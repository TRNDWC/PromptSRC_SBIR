"""Building blocks for the PromptSRC-style prompt learning used by
``src.model_promptsrc``.

Three pieces live here:
  * ``VisualPromptLearner`` -- deep visual prompts for one branch (photo / sketch).
  * ``TextPromptLearner``   -- deep text prompts + the handcrafted template pool
                               used to build the frozen text anchors.
  * ``GaussianPromptAggregator`` -- GPA (PromptSRC Eq. 6-7).

Nothing here hardcodes a hyper-parameter: J, V, T, N, mu, sigma^2 all come from
the config (``experiments/options.py``).
"""

from contextlib import contextmanager

import torch
import torch.nn as nn

from src.clip import clip

# Handcrafted template pools used to build the FROZEN text anchors (the
# "textual diversity" of PromptSRC).  The paper ensembles N=60 ImageNet
# templates; how many of these are actually used is controlled by
# ``--n_text_templates``.  These are data (a vocabulary), not tuned
# hyper-parameters -- the count N is the hyper-parameter and it is in the config.
PHOTO_TEMPLATES = [
    'a photo of a {}.',
    'a photo of the {}.',
    'a bright photo of a {}.',
    'a cropped photo of a {}.',
    'a close-up photo of a {}.',
    'a good photo of a {}.',
    'a photo of one {}.',
    'a low resolution photo of a {}.',
    'a blurry photo of a {}.',
    'a photo of a large {}.',
    'a photo of a small {}.',
    'itap of a {}.',
]

SKETCH_TEMPLATES = [
    'a sketch of a {}.',
    'a drawing of a {}.',
    'a sketch of the {}.',
    'a black and white sketch of a {}.',
    'a rough sketch of a {}.',
    'a hand drawn sketch of a {}.',
    'a doodle of a {}.',
    'a line drawing of a {}.',
    'a pencil sketch of a {}.',
    'a simple sketch of a {}.',
    'a quick sketch of a {}.',
    'a sketch of one {}.',
]

TEMPLATE_POOLS = {'photo': PHOTO_TEMPLATES, 'sketch': SKETCH_TEMPLATES}


class VisualPromptLearner(nn.Module):
    """J layers x V tokens of learnable visual prompt for a single branch.

    Layer 0 is appended to the patch tokens at the encoder input (this reuses the
    existing shallow-prompt path of ``VisionTransformer.forward``); layers
    1..J-1 overwrite those same token positions at the input of the
    corresponding transformer block.
    """

    def __init__(self, depth: int, n_tokens: int, width: int, init_std: float = 0.02):
        super().__init__()
        assert depth >= 1 and n_tokens >= 1
        self.depth = depth
        self.n_tokens = n_tokens
        self.width = width
        self.prompts = nn.Parameter(torch.empty(depth, n_tokens, width))
        nn.init.normal_(self.prompts, std=init_std)

    def first(self, batch_size: int) -> torch.Tensor:
        """Layer-0 prompt, expanded to [B, V, width]."""
        return self.prompts[0].unsqueeze(0).expand(batch_size, -1, -1)

    def deep(self):
        """Prompts for blocks 1..J-1, or None when J == 1 (shallow prompting)."""
        if self.depth == 1:
            return None
        return [self.prompts[i] for i in range(1, self.depth)]


class TextPromptLearner(nn.Module):
    """Learnable text context, optionally one set per domain.

    ``mode='shared'``     -> a single P_t used by both the photo- and the
                             sketch-side text branch (the design we start from).
    ``mode='per_domain'`` -> P_t_photo and P_t_sketch, separate weights, so the
                             two-set variant can be tried by flipping a flag.
    """

    def __init__(self, depth, n_tokens, width, mode='shared',
                 init_text=None, token_embedding=None, init_std=0.02):
        super().__init__()
        assert mode in ('shared', 'per_domain')
        assert depth >= 1 and n_tokens >= 1
        self.depth = depth
        self.n_tokens = n_tokens
        self.mode = mode
        init_text = init_text or {}

        self.domains = ['shared'] if mode == 'shared' else ['photo', 'sketch']
        self.ctx = nn.ParameterDict()
        for d in self.domains:
            p = nn.Parameter(torch.empty(depth, n_tokens, width))
            nn.init.normal_(p, std=init_std)
            self.ctx[d] = p

        # Initialise the layer-0 context from a handcrafted template when the
        # token count lines up (CoOp-style init; falls back to random otherwise).
        if token_embedding is not None:
            for d in self.domains:
                text = init_text.get(d)
                if not text:
                    continue
                with torch.no_grad():
                    tok = clip.tokenize(text)
                    emb = token_embedding(tok.to(token_embedding.weight.device))[0]
                    n_words = int(tok.argmax(dim=-1)[0]) - 1  # tokens between SOS and EOT
                    if n_words >= n_tokens:
                        self.ctx[d][0].copy_(emb[1:1 + n_tokens].float())

        # placeholder tokens that the learnable context overwrites
        self._placeholder = ' '.join(['X'] * n_tokens)
        self._token_cache = {}

    def _key(self, domain):
        return 'shared' if self.mode == 'shared' else domain

    def ctx_first(self, domain):
        return self.ctx[self._key(domain)][0]

    def ctx_deep(self, domain):
        p = self.ctx[self._key(domain)]
        if self.depth == 1:
            return None
        return [p[i] for i in range(1, self.depth)]

    def tokenize(self, classnames, device):
        """Tokenised '<X X ... X> {classname}.' prompts, cached per classname."""
        missing = [c for c in classnames if c not in self._token_cache]
        if missing:
            texts = ['%s %s.' % (self._placeholder, c.replace('_', ' ')) for c in missing]
            toks = clip.tokenize(texts)
            for c, t in zip(missing, toks):
                self._token_cache[c] = t
        return torch.stack([self._token_cache[c] for c in classnames]).to(device)


class GaussianPromptAggregator(nn.Module):
    """Gaussian weighted prompt aggregation, PromptSRC Eq. 6-7.

    Keeps ONE running copy per parameter instead of the whole prompt history:
    after epoch e we do ``gpa <- gpa + w_e * P_e`` with a Gaussian weight
    ``w_e ~ exp(-(e-mu)^2 / (2 sigma^2))`` normalised over all epochs.  Reading
    the aggregate divides by the accumulated weight, so a run stopped early
    still yields a proper convex combination of the epochs it did see.
    """

    def __init__(self, module: nn.Module, max_epochs: int, mu: float, sigma: float):
        super().__init__()
        assert max_epochs >= 1 and sigma > 0
        e = torch.arange(max_epochs, dtype=torch.float32)
        w = torch.exp(-((e - mu) ** 2) / (2.0 * sigma ** 2))
        self.register_buffer('weights', w / w.sum())
        self.register_buffer('acc_w', torch.zeros(()))
        self._names = []
        for name, p in module.named_parameters():
            buf = name.replace('.', '__')
            self.register_buffer(buf, torch.zeros_like(p))
            self._names.append((name, buf))

    @torch.no_grad()
    def update(self, module: nn.Module, epoch: int):
        idx = min(int(epoch), self.weights.numel() - 1)
        w = self.weights[idx]
        params = dict(module.named_parameters())
        for name, buf in self._names:
            getattr(self, buf).add_(w * params[name].detach().to(getattr(self, buf).dtype))
        self.acc_w += w

    def is_ready(self) -> bool:
        return bool(self.acc_w.item() > 0)

    @torch.no_grad()
    def aggregated(self):
        scale = 1.0 / max(self.acc_w.item(), 1e-12)
        return {name: getattr(self, buf) * scale for name, buf in self._names}


@contextmanager
def use_aggregated_prompts(pairs):
    """Temporarily swap each module's parameters for its GPA aggregate.

    ``pairs`` is an iterable of (module, aggregator).  Aggregators that have not
    accumulated anything yet (epoch 0, or GPA disabled) are skipped, so the
    caller always gets a working forward pass.
    """
    backups = []
    try:
        for module, agg in pairs:
            if agg is None or not agg.is_ready():
                continue
            params = dict(module.named_parameters())
            agg_vals = agg.aggregated()
            for name, value in agg_vals.items():
                p = params[name]
                backups.append((p, p.detach().clone()))
                with torch.no_grad():
                    p.copy_(value.to(p.dtype))
        yield
    finally:
        with torch.no_grad():
            for p, old in backups:
                p.copy_(old)
