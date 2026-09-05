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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.functional import retrieval_average_precision, retrieval_precision

from src.clip import clip
from src.clip.model import convert_weights
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
        self.val_step_outputs_sk = []
        self.val_step_outputs_ph = []

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

    def configure_optimizers(self):
        # CLIP is fully frozen, so only the prompts are optimised.
        return torch.optim.Adam(self.prompt_parameters(), lr=self.opts.prompt_lr)

    # ------------------------------------------------------------- image side
    def encode_image_branch(self, images, domain, return_anchor=True):
        """Returns (prompted_feature, frozen_anchor_feature).

        The anchor is computed with NO prompt and under ``torch.no_grad()``, so
        no gradient can leak into it (see the sanity check).
        """
        learner = self.prompt_photo if domain == 'photo' else self.prompt_sketch
        feat = self.clip.encode_image(
            images, learner.first(images.shape[0]), learner.deep())
        anchor = None
        if return_anchor:
            with torch.no_grad():
                anchor = self.clip.encode_image(images).detach()
        return feat, anchor

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
        out = []
        for name in classnames:
            key = (domain, name, len(templates))
            if key not in self._anchor_text_cache:
                texts = [t.format(name.replace('_', ' ')) for t in templates]
                tok = clip.tokenize(texts).to(self.device)
                feats = self.clip.encode_text(tok)
                feats = F.normalize(feats.float(), dim=-1).mean(dim=0)
                self._anchor_text_cache[key] = F.normalize(feats, dim=-1).detach().cpu()
            out.append(self._anchor_text_cache[key])
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

    def _logits(self, img_feat, txt_feat):
        img = F.normalize(img_feat.float(), dim=-1)
        txt = F.normalize(txt_feat.float(), dim=-1)
        return self.clip.logit_scale.exp().float() * img @ txt.t()

    def compute_losses(self, sk_tensor, img_tensor, neg_tensor, categories):
        """All PromptSRC-for-SBIR losses, returned separately for logging."""
        o = self.opts
        categories = list(categories)

        photo_p, photo_a = self.encode_image_branch(img_tensor, 'photo')
        sketch_p, sketch_a = self.encode_image_branch(sk_tensor, 'sketch')
        # negatives are photos -> P_v_photo, but they are NOT regularised by SCL
        neg_p, _ = self.encode_image_branch(neg_tensor, 'photo', return_anchor=False)

        # class vocabulary of this batch, used for the SCL-logits term
        class_list = sorted(set(categories))
        cls_index = torch.tensor([class_list.index(c) for c in categories], device=self.device)

        text_p, text_a = {}, {}
        for domain in sorted(set(self.text_scl_domains + [o.scl_logits_anchor])):
            text_p[domain] = self.encode_text_prompted(class_list, domain)
            text_a[domain] = self.encode_text_anchor(class_list, domain)

        losses = {}
        losses['L_retrieval'] = self.loss_fn(sketch_p, photo_p, neg_p)
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

        losses['loss'] = (losses['L_retrieval']
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
            self.log('train/%s' % name, value, prog_bar=(name == 'loss'))
        self.log('train_loss', losses['loss'])
        return losses['loss']

    def on_train_epoch_end(self):
        # GPA moving average, one update per epoch (Eq. 6-7)
        for module, agg in self._gpa_pairs():
            if agg is not None:
                agg.update(module, self.current_epoch)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        image_tensor, label = batch
        domain = 'sketch' if dataloader_idx == 0 else 'photo'
        # inference uses the GPA prompts, not the last-epoch prompts
        with use_aggregated_prompts(self._gpa_pairs()), torch.no_grad():
            feat, _ = self.encode_image_branch(image_tensor, domain, return_anchor=False)
        if dataloader_idx == 0:
            self.val_step_outputs_sk.append((feat.float(), label))
        else:
            self.val_step_outputs_ph.append((feat.float(), label))

    def on_validation_epoch_end(self):
        if not self.val_step_outputs_sk or not self.val_step_outputs_ph:
            self.val_step_outputs_sk.clear()
            self.val_step_outputs_ph.clear()
            return

        query = torch.cat([f for f, _ in self.val_step_outputs_sk])
        gallery = torch.cat([f for f, _ in self.val_step_outputs_ph])
        sk_cat = np.concatenate([l.cpu().numpy() for _, l in self.val_step_outputs_sk])
        ph_cat = np.concatenate([l.cpu().numpy() for _, l in self.val_step_outputs_ph])

        map_k = p_k = 200
        ap = torch.zeros(len(query))
        precision = torch.zeros(len(query))
        for idx, sk_feat in enumerate(query):
            sim = F.cosine_similarity(sk_feat.unsqueeze(0), gallery).cpu()
            target = torch.zeros(len(gallery), dtype=torch.bool)
            target[np.where(ph_cat == sk_cat[idx])] = True
            ap[idx] = retrieval_average_precision(sim, target, top_k=min(map_k, len(gallery)))
            precision[idx] = retrieval_precision(sim, target, top_k=min(p_k, len(gallery)))

        mAP = torch.mean(ap)
        self.log('mAP', mAP, on_step=False, on_epoch=True)
        self.log('prec', torch.mean(precision), on_step=False, on_epoch=True)
        self.best_metric = max(self.best_metric, mAP.item())
        print('mAP@%d: %.4f, P@%d: %.4f, best mAP: %.4f'
              % (map_k, mAP.item(), p_k, torch.mean(precision).item(), self.best_metric))

        self.val_step_outputs_sk.clear()
        self.val_step_outputs_ph.clear()
