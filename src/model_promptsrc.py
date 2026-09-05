"""PromptSRC-style prompt learning adapted to two-branch ZS-SBIR.

Differences from the original PromptSRC (Khattak et al., ICCV 2023), all driven
by the retrieval setting:
  * two visual prompt sets (P_v_photo, P_v_sketch) with the SAME J/V but their
    own weights, both going through the SAME frozen visual encoder -- there is
    only ONE copy of CLIP in memory;
  * the main objective stays the triplet retrieval loss already used by the
    basecode (``src/model_LN_prompt.py``), not a classification CE;
  * self-regularisation (SCL) is applied to both image branches, to the text
    branch, and to the sketch-vs-text logits.

The old ``src/model_LN_prompt.py`` is left untouched; this is a parallel model.
"""

from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.functional import retrieval_average_precision, retrieval_precision

from src.clip import clip
from src.clip.model import convert_weights
from src.dataset_retrieval import get_metric_config
from src.promptsrc_modules import (
    TEMPLATE_POOLS,
    GaussianPromptAggregator,
    TextPromptLearner,
    VisualPromptLearner,
    use_aggregated_prompts,
)
from experiments.options import opts as default_opts


def freeze_model(m):
    m.requires_grad_(False)


class PromptSRCModel(pl.LightningModule):

    def __init__(self, opts=None):
        super().__init__()
        self.opts = opts if opts is not None else default_opts
        o = self.opts

        # ---- frozen CLIP (single copy, shared by both branches) -------------
        self.clip, _ = clip.load('ViT-B/32', device='cpu')  # cpu load => fp32
        if not int(o.clip_float):
            convert_weights(self.clip)  # back to fp16 if explicitly asked for
        self.clip.apply(freeze_model)
        self.clip.eval()

        # ---- optionally unfreeze every LayerNorm (both encoders) -------------
        # This is the "LN_prompt" idea the basecode names but never enables
        # (freeze_all_but_bn in src/model_LN_prompt.py is defined and unused).
        self._ln_params = []
        self._ln_backup = []
        if int(o.train_layernorm):
            for mod in self.clip.modules():
                if isinstance(mod, nn.LayerNorm):
                    for p in mod.parameters():
                        p.requires_grad_(True)
                        self._ln_params.append(p)
            # Pristine copy of the LN weights. The SCL anchors are supposed to be
            # the ORIGINAL pre-trained CLIP; once LN is trainable the encoder
            # drifts, so without this the anchor would be a moving target and the
            # regularisation would collapse towards a no-op. Restored around
            # every anchor forward -- see original_ln().
            for name, p in self.clip.named_parameters():
                if p.requires_grad:
                    buf = 'ln0__' + name.replace('.', '__')
                    self.register_buffer(buf, p.detach().clone())
                    self._ln_backup.append((name, buf))

        vision_width = self.clip.visual.conv1.out_channels
        text_width = self.clip.transformer.width

        # ---- prompts ---------------------------------------------------------
        self.prompt_photo = VisualPromptLearner(
            o.prompt_depth_vision, o.n_prompts_vision, vision_width, o.prompt_init_std)
        self.prompt_sketch = VisualPromptLearner(
            o.prompt_depth_vision, o.n_prompts_vision, vision_width, o.prompt_init_std)
        self.prompt_text = TextPromptLearner(
            o.prompt_depth_text, o.n_prompts_text, text_width,
            mode=o.text_prompt_mode,
            init_text={'shared': o.text_ctx_init_photo,
                       'photo': o.text_ctx_init_photo,
                       'sketch': o.text_ctx_init_sketch},
            token_embedding=self.clip.token_embedding,
            init_std=o.prompt_init_std)

        # ---- GPA: one aggregator per branch, own mu / sigma^2 ---------------
        self.gpa_photo = self.gpa_sketch = self.gpa_text = None
        if int(o.gpa_enabled):
            self.gpa_photo = GaussianPromptAggregator(
                self.prompt_photo, o.max_epochs, o.gpa_mu_photo, o.gpa_sigma2_photo ** 0.5)
            self.gpa_sketch = GaussianPromptAggregator(
                self.prompt_sketch, o.max_epochs, o.gpa_mu_sketch, o.gpa_sigma2_sketch ** 0.5)
            if int(o.gpa_apply_text):
                self.gpa_text = GaussianPromptAggregator(
                    self.prompt_text, o.max_epochs, o.gpa_mu_text, o.gpa_sigma2_text ** 0.5)

        # ---- losses ----------------------------------------------------------
        # unchanged from the basecode: cosine triplet, margin 0.2
        self.distance_fn = lambda x, y: 1.0 - F.cosine_similarity(x, y)
        self.loss_fn = nn.TripletMarginWithDistanceLoss(
            distance_function=self.distance_fn, margin=0.2)

        self.text_scl_domains = [d.strip() for d in o.text_scl_domains.split(',') if d.strip()]
        self._anchor_text_cache = {}

        self.best_metric = -1e3
        self.val_step_outputs = []

    def train(self, mode=True):
        # the backbone stays in eval mode whatever Lightning does to the module
        super().train(mode)
        self.clip.eval()
        return self

    # ------------------------------------------------------------------ utils
    def _gpa_pairs(self):
        return [(self.prompt_photo, self.gpa_photo),
                (self.prompt_sketch, self.gpa_sketch),
                (self.prompt_text, self.gpa_text)]

    def prompt_parameters(self):
        return (list(self.prompt_photo.parameters())
                + list(self.prompt_sketch.parameters())
                + list(self.prompt_text.parameters()))

    @contextmanager
    def original_ln(self):
        """Restore the pre-trained LayerNorm weights for the duration of the block.

        No-op when LN training is off, or when --anchor_uses_original_ln=0 (in
        which case the anchors follow the drifting encoder on purpose).
        """
        if not self._ln_backup or not int(self.opts.anchor_uses_original_ln):
            yield
            return
        params = dict(self.clip.named_parameters())
        saved = []
        try:
            with torch.no_grad():
                for name, buf in self._ln_backup:
                    p = params[name]
                    saved.append((p, p.detach().clone()))
                    p.copy_(getattr(self, buf))
            yield
        finally:
            with torch.no_grad():
                for p, old in saved:
                    p.copy_(old)

    def configure_optimizers(self):
        # Prompts always; LayerNorm too when --train_layernorm=1, at its own lr.
        groups = [{'params': self.prompt_parameters(), 'lr': self.opts.prompt_lr}]
        if self._ln_params:
            groups.append({'params': self._ln_params, 'lr': self.opts.clip_LN_lr})
        return torch.optim.Adam(groups)

    # ------------------------------------------------------------- image side
    def encode_image_prompted(self, images, domain):
        learner = self.prompt_photo if domain == 'photo' else self.prompt_sketch
        return self.clip.encode_image(
            images, learner.first(images.shape[0]), learner.deep())

    @torch.no_grad()
    def encode_image_anchor(self, images):
        """Frozen anchor: no prompt, no gradient.

        MUST be called inside ``self.original_ln()`` when LayerNorm is trainable
        -- and BEFORE any prompted forward whose graph is still alive, see
        compute_losses().
        """
        return self.clip.encode_image(images).detach()

    def encode_image_branch(self, images, domain, return_anchor=True):
        """Returns (prompted_feature, frozen_anchor_feature).

        Anchor first, then the prompted pass: original_ln() writes the pristine
        LayerNorm weights in place, which bumps their autograd version counter
        and would invalidate an already-built prompted graph.
        """
        anchor = None
        if return_anchor:
            with self.original_ln():
                anchor = self.encode_image_anchor(images)
        return self.encode_image_prompted(images, domain), anchor

    # -------------------------------------------------------------- text side
    def encode_text_prompted(self, classnames, domain):
        tokens = self.prompt_text.tokenize(classnames, self.device)
        return self.clip.encode_text(
            tokens, self.prompt_text.ctx_first(domain), self.prompt_text.ctx_deep(domain))

    @torch.no_grad()
    def encode_text_anchor(self, classnames, domain):
        """Frozen text anchor = mean over N handcrafted templates.

        Each template embedding is L2-normalised before averaging (standard CLIP
        prompt ensembling), then the mean is re-normalised.
        """
        templates = TEMPLATE_POOLS[domain][:int(self.opts.n_text_templates)]
        assert templates, 'n_text_templates must be >= 1'
        # Caching is only sound while the text encoder is fixed. With trainable
        # LayerNorm and --anchor_uses_original_ln=0 the anchor changes every
        # step, so the cache is bypassed rather than silently serving stale
        # features.
        use_cache = not self._ln_params or int(self.opts.anchor_uses_original_ln)
        out = []
        with self.original_ln():
            for name in classnames:
                key = (domain, name, len(templates))
                if use_cache and key in self._anchor_text_cache:
                    out.append(self._anchor_text_cache[key])
                    continue
                texts = [t.format(name.replace('_', ' ')) for t in templates]
                tok = clip.tokenize(texts).to(self.device)
                feats = self.clip.encode_text(tok)
                feats = F.normalize(feats.float(), dim=-1).mean(dim=0)
                feats = F.normalize(feats, dim=-1).detach().cpu()
                if use_cache:
                    self._anchor_text_cache[key] = feats
                out.append(feats)
        return torch.stack(out).to(self.device).detach()

    # ------------------------------------------------------------------ losses
    def _scl_feature_loss(self, prompted, anchor):
        """L1 between a prompted feature and its frozen anchor.

        Both sides are L2-normalised when ``--scl_normalize_features`` is on
        (default): the text anchor is an ensemble mean of normalised embeddings
        and therefore lives on a different scale than a raw CLIP feature, so an
        L1 on raw vectors would be dominated by that scale gap.
        """
        p, a = prompted.float(), anchor.float()
        if int(getattr(self.opts, 'scl_normalize_features', 1)):
            p, a = F.normalize(p, dim=-1), F.normalize(a, dim=-1)
        return F.l1_loss(p, a)

    def _infonce(self, sk_feat, ph_feat, cls_index):
        """Symmetric InfoNCE between the prompted sketch and photo features.

        In-batch negatives, temperature from --infonce_temperature.

        --infonce_mode=instance is textbook InfoNCE: only (sketch_i, photo_i) is
        positive. Careful -- the dataloader pairs each sketch with a RANDOM photo
        of its own category, so two samples of the same class in one batch make
        each other false negatives, and the loss then actively pushes apart
        photos and sketches that should match at category level. With ~104 train
        classes and a batch of 64 that happens in most batches.
        --infonce_mode=class treats every same-category pair as positive
        (multi-positive InfoNCE) and removes that conflict.
        """
        sk = F.normalize(sk_feat.float(), dim=-1)
        ph = F.normalize(ph_feat.float(), dim=-1)
        logits = sk @ ph.t() / float(self.opts.infonce_temperature)

        if self.opts.infonce_mode == 'instance':
            target = torch.arange(sk.shape[0], device=logits.device)
            return 0.5 * (F.cross_entropy(logits, target)
                          + F.cross_entropy(logits.t(), target))

        pos = (cls_index[:, None] == cls_index[None, :]).float()
        def _multi_pos(lg):
            log_prob = F.log_softmax(lg, dim=-1)
            return -((log_prob * pos).sum(dim=-1) / pos.sum(dim=-1).clamp(min=1)).mean()
        return 0.5 * (_multi_pos(logits) + _multi_pos(logits.t()))

    def _logits(self, img_feat, txt_feat):
        img = F.normalize(img_feat.float(), dim=-1)
        txt = F.normalize(txt_feat.float(), dim=-1)
        return self.clip.logit_scale.exp().float() * img @ txt.t()

    def compute_losses(self, sk_tensor, img_tensor, neg_tensor, categories):
        """All PromptSRC-for-SBIR losses, returned separately for logging."""
        o = self.opts
        categories = list(categories)

        # class vocabulary of this batch, used for the SCL-logits term
        class_list = sorted(set(categories))
        cls_index = torch.tensor([class_list.index(c) for c in categories], device=self.device)
        domains = sorted(set(self.text_scl_domains + [o.scl_logits_anchor]))

        # ---- every frozen anchor FIRST, in one original-LayerNorm block ------
        # Ordering matters: original_ln() restores the pristine LN weights in
        # place, so it has to run before any prompted graph exists, otherwise
        # autograd rejects the backward ("variable needed for gradient
        # computation has been modified by an inplace operation").
        with self.original_ln():
            photo_a = self.encode_image_anchor(img_tensor)
            sketch_a = self.encode_image_anchor(sk_tensor)
            text_a = {d: self.encode_text_anchor(class_list, d) for d in domains}

        # ---- prompted branches (these build the graph) -----------------------
        photo_p = self.encode_image_prompted(img_tensor, 'photo')
        sketch_p = self.encode_image_prompted(sk_tensor, 'sketch')
        # negatives are photos -> P_v_photo, but they are NOT regularised by SCL
        neg_p = self.encode_image_prompted(neg_tensor, 'photo')
        text_p = {d: self.encode_text_prompted(class_list, d) for d in domains}

        losses = {}
        losses['L_retrieval'] = self.loss_fn(sketch_p, photo_p, neg_p)
        losses['L_infonce'] = self._infonce(sketch_p, photo_p, cls_index)
        losses['L_SCL_image_photo'] = self._scl_feature_loss(photo_p, photo_a)
        losses['L_SCL_image_sketch'] = self._scl_feature_loss(sketch_p, sketch_a)
        losses['L_SCL_text'] = torch.stack([
            self._scl_feature_loss(text_p[d], text_a[d]) for d in self.text_scl_domains
        ]).mean()

        # DESIGN HYPOTHESIS: which image branch pairs with the text logits is a
        # choice, not something the paper settles for retrieval -- hence the
        # --scl_logits_anchor flag (sketch by default) instead of a hardcoded one.
        anchor_domain = o.scl_logits_anchor
        img_p = sketch_p if anchor_domain == 'sketch' else photo_p
        img_a = sketch_a if anchor_domain == 'sketch' else photo_a
        logits_p = self._logits(img_p, text_p[anchor_domain])
        logits_a = self._logits(img_a, text_a[anchor_domain])
        # frozen pair is the teacher: KL(frozen || prompted)
        losses['L_SCL_logits'] = F.kl_div(
            F.log_softmax(logits_p, dim=-1),
            F.log_softmax(logits_a, dim=-1),
            reduction='batchmean', log_target=True)

        losses['loss'] = (o.lambda_retrieval * losses['L_retrieval']
                          + o.lambda_infonce * losses['L_infonce']
                          + o.lambda_scl_image * (losses['L_SCL_image_photo']
                                                  + losses['L_SCL_image_sketch'])
                          + o.lambda_scl_text * losses['L_SCL_text']
                          + o.lambda_scl_logits * losses['L_SCL_logits'])

        feats = {'photo_prompted': photo_p, 'photo_anchor': photo_a,
                 'sketch_prompted': sketch_p, 'sketch_anchor': sketch_a,
                 'neg_prompted': neg_p,
                 'text_prompted': text_p, 'text_anchor': text_a,
                 'logits_prompted': logits_p, 'logits_anchor': logits_a,
                 'cls_index': cls_index}
        return losses, feats

    # ------------------------------------------------------------------- steps
    def training_step(self, batch, batch_idx):
        sk_tensor, img_tensor, neg_tensor, category = batch[:4]
        losses, _ = self.compute_losses(sk_tensor, img_tensor, neg_tensor, category)
        for name, value in losses.items():
            self.log('train/%s' % name, value)
        self.log('train_loss', losses['loss'])

        # Compact live breakdown: the total alone cannot tell "retrieval is
        # improving" apart from "the SCL terms collapsed", and with lambda1=10 /
        # lambda2=25 the SCL terms carry most of the total.
        self.log_dict({
            'loss': losses['loss'],
            'ret': losses['L_retrieval'],
            'nce': losses['L_infonce'],
            'scl_i': losses['L_SCL_image_photo'] + losses['L_SCL_image_sketch'],
            'scl_t': losses['L_SCL_text'],
            'scl_lg': losses['L_SCL_logits'],
        }, prog_bar=True, logger=False)
        return losses['loss']

    def on_train_epoch_end(self):
        # GPA moving average, one update per epoch (Eq. 6-7)
        for module, agg in self._gpa_pairs():
            if agg is not None:
                agg.update(module, self.current_epoch)

    def validation_step(self, batch, batch_idx):
        """Same protocol as the original repo (aneeshan95/Sketch_LVM): the val
        set is Sketchy(mode='val') over the unseen classes, the query set is the
        sketch features and the gallery is the paired photo features."""
        sk_tensor, img_tensor, neg_tensor, category = batch[:4]
        # inference uses the GPA prompts, not the last-epoch prompts
        with use_aggregated_prompts(self._gpa_pairs()), torch.no_grad():
            sk_feat, _ = self.encode_image_branch(sk_tensor, 'sketch', return_anchor=False)
            img_feat, _ = self.encode_image_branch(img_tensor, 'photo', return_anchor=False)
            neg_feat, _ = self.encode_image_branch(neg_tensor, 'photo', return_anchor=False)
        self.log('val_loss', self.loss_fn(sk_feat, img_feat, neg_feat))
        self.val_step_outputs.append((sk_feat.float(), img_feat.float(), list(category)))

    def on_validation_epoch_end(self):
        if not self.val_step_outputs:
            return

        query = torch.cat([q for q, _, _ in self.val_step_outputs])
        gallery = torch.cat([g for _, g, _ in self.val_step_outputs])
        all_category = np.array(sum([c for _, _, c in self.val_step_outputs], []))

        # Original metric: mAP over the whole ranking, no cutoff. mAP@k and P@k
        # follow the per-dataset protocol in dataset_retrieval.DATASET_METRICS
        # (Sketchy 200/200, TU-Berlin all/100, QuickDraw all/200).
        map_k, p_k = get_metric_config(self.opts)
        ap_all = torch.zeros(len(query))
        ap_k = torch.zeros(len(query))
        precision = torch.zeros(len(query))
        for idx, sk_feat in enumerate(query):
            # The original ranks by -1 * (1 - cos), i.e. cos - 1, which is <= 0
            # everywhere. That is correct on torchmetrics 0.9.3 (pinned in
            # environment.yml), which only argsorts. Newer torchmetrics added
            # `target = torch.where(preds > 0, target, 0)` to
            # retrieval_average_precision/precision/recall/reciprocal_rank, so
            # every non-positive score is forced to "irrelevant" and the metric
            # collapses to exactly 0.0. Rank by (1 + cos) / 2 instead: same
            # ordering, strictly positive, correct on both versions.
            cos = 1.0 - self.distance_fn(sk_feat.unsqueeze(0), gallery)
            score = ((1.0 + cos) / 2.0).clamp(min=1e-6).cpu()
            target = torch.zeros(len(gallery), dtype=torch.bool)
            target[np.where(all_category == all_category[idx])] = True
            ap_all[idx] = retrieval_average_precision(score, target)
            ap_k[idx] = retrieval_average_precision(
                score, target, top_k=min(map_k, len(gallery))) if map_k > 0 else ap_all[idx]
            precision[idx] = retrieval_precision(
                score, target, top_k=min(p_k, len(gallery)) if p_k > 0 else None)

        # 'mAP' is the protocol metric of --dataset (mAP@200 for Sketchy,
        # mAP@all for TU-Berlin / QuickDraw). ModelCheckpoint and best_metric
        # track it, so the selected checkpoint is the best under the protocol
        # actually being reported. mAP@all is logged too when they differ.
        mAP = torch.mean(ap_k) if map_k > 0 else torch.mean(ap_all)
        self.log('mAP', mAP, on_step=False, on_epoch=True)
        self.log('mAP_all', torch.mean(ap_all), on_step=False, on_epoch=True)
        self.log('prec', torch.mean(precision), on_step=False, on_epoch=True)
        self.best_metric = max(self.best_metric, mAP.item())

        map_name = 'mAP@%d' % map_k if map_k > 0 else 'mAP@all'
        p_name = 'P@%d' % p_k if p_k > 0 else 'P@all'
        parts = ['%s: %.4f' % (map_name, mAP.item()),
                 '%s: %.4f' % (p_name, torch.mean(precision).item())]
        if map_k > 0:  # secondary, not the protocol number
            parts.append('(mAP@all: %.4f)' % torch.mean(ap_all).item())
        print('[%s] %s | best %s: %.4f  (|query|=%d, |gallery|=%d)'
              % (self.opts.dataset, ' | '.join(parts), map_name, self.best_metric,
                 len(query), len(gallery)))

        self.val_step_outputs.clear()
