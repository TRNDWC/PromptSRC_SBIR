"""Sanity check for the PromptSRC-for-SBIR pipeline (BƯỚC 4).

Runs one batch_size=2 forward/backward on random tensors -- no dataset needed --
and reports feature shapes, each loss separately, gradient isolation of the
frozen anchor branch, backward-compatibility of the patched CLIP, and GPA
correctness.

    python -m experiments.sanity_promptsrc [--max_epochs=4 ...]
"""

import torch

from experiments.options import opts
from src.clip import clip
from src.model_promptsrc import PromptSRCModel
from src.promptsrc_modules import use_aggregated_prompts

OK, BAD = '  [ok]  ', '  [FAIL]'
failures = []


def check(name, condition, detail=''):
    print('%s %s %s' % (OK if condition else BAD, name, detail))
    if not condition:
        failures.append(name)


def main():
    torch.manual_seed(0)
    device = 'cpu'

    # small config so the check is fast; every value comes from the config object
    opts.max_epochs = max(int(opts.max_epochs), 4)
    print('config: J_v=%d V=%d | J_t=%d T=%d | N_templates=%d | text_prompt_mode=%s\n'
          '        lambda1=%.1f lambda2=%.1f lambda_logits=%.1f | scl_logits_anchor=%s\n'
          '        gpa: photo(mu=%.1f,s2=%.1f) sketch(mu=%.1f,s2=%.1f) text(mu=%.1f,s2=%.1f) max_epochs=%d'
          % (opts.prompt_depth_vision, opts.n_prompts_vision,
             opts.prompt_depth_text, opts.n_prompts_text,
             opts.n_text_templates, opts.text_prompt_mode,
             opts.lambda_scl_image, opts.lambda_scl_text, opts.lambda_scl_logits,
             opts.scl_logits_anchor,
             opts.gpa_mu_photo, opts.gpa_sigma2_photo,
             opts.gpa_mu_sketch, opts.gpa_sigma2_sketch,
             opts.gpa_mu_text, opts.gpa_sigma2_text, opts.max_epochs))

    print('\n== building model ==')
    model = PromptSRCModel(opts).to(device)
    model.train()

    n_prompt = sum(p.numel() for p in model.prompt_parameters())
    n_clip = sum(p.numel() for p in model.clip.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('  CLIP params: %d (frozen) | prompt params: %d | trainable: %d'
          % (n_clip, n_prompt, n_train))
    check('only prompts are trainable', n_train == n_prompt, '(%d == %d)' % (n_train, n_prompt))
    check('CLIP is fp32 (clip_float)', model.clip.dtype == torch.float32, str(model.clip.dtype))

    # ---------------------------------------------------------------- 0. compat
    print('\n== 0. backward-compatibility of the patched CLIP ==')
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        plain_img = model.clip.encode_image(x)
        plain_txt = model.clip.encode_text(clip.tokenize(['a photo of a cow.', 'a sketch of a tree.']))
    check('encode_image(x) with no prompt runs', tuple(plain_img.shape) == (2, 512), str(tuple(plain_img.shape)))
    check('encode_text(t) with no ctx runs', tuple(plain_txt.shape) == (2, 512), str(tuple(plain_txt.shape)))

    # ------------------------------------------------------------- 1. forward
    print('\n== 1. forward, batch_size=2 ==')
    sk = torch.randn(2, 3, 224, 224)
    img = torch.randn(2, 3, 224, 224)
    neg = torch.randn(2, 3, 224, 224)
    categories = ['cow', 'tree']

    losses, feats = model.compute_losses(sk, img, neg, categories)

    print('  feature shapes:')
    for key in ['photo_prompted', 'photo_anchor', 'sketch_prompted', 'sketch_anchor', 'neg_prompted']:
        print('    %-18s %s  requires_grad=%s' % (key, tuple(feats[key].shape), feats[key].requires_grad))
    for domain, t in feats['text_prompted'].items():
        print('    %-18s %s  requires_grad=%s' % ('text_prompted[%s]' % domain, tuple(t.shape), t.requires_grad))
    for domain, t in feats['text_anchor'].items():
        print('    %-18s %s  requires_grad=%s' % ('text_anchor[%s]' % domain, tuple(t.shape), t.requires_grad))
    print('    %-18s %s' % ('logits_prompted', tuple(feats['logits_prompted'].shape)))
    print('    %-18s %s' % ('logits_anchor', tuple(feats['logits_anchor'].shape)))

    check('prompted image feats have grad',
          feats['photo_prompted'].requires_grad and feats['sketch_prompted'].requires_grad)
    check('prompted text feats have grad',
          all(t.requires_grad for t in feats['text_prompted'].values()))

    # ------------------------------------------------- 2. anchor has no grad
    print('\n== 2. anchor branch gradient isolation ==')
    for key in ['photo_anchor', 'sketch_anchor']:
        check('%s.requires_grad is False' % key, feats[key].requires_grad is False)
    for domain, t in feats['text_anchor'].items():
        check('text_anchor[%s].requires_grad is False' % domain, t.requires_grad is False)
    check('logits_anchor.requires_grad is False', feats['logits_anchor'].requires_grad is False)

    # ------------------------------------------------------------- 3. losses
    print('\n== 3. individual losses ==')
    for name in ['L_retrieval', 'L_SCL_image_photo', 'L_SCL_image_sketch',
                 'L_SCL_text', 'L_SCL_logits', 'loss']:
        v = losses[name].item()
        print('    %-20s = %.6f' % (name, v))
        check('%s is finite' % name, torch.isfinite(losses[name]).item(), '')
    for name in ['L_SCL_image_photo', 'L_SCL_image_sketch', 'L_SCL_text', 'L_SCL_logits']:
        check('%s >= 0' % name, losses[name].item() >= 0.0, '(%.6f)' % losses[name].item())

    expected = (losses['L_retrieval']
                + opts.lambda_scl_image * (losses['L_SCL_image_photo'] + losses['L_SCL_image_sketch'])
                + opts.lambda_scl_text * losses['L_SCL_text']
                + opts.lambda_scl_logits * losses['L_SCL_logits'])
    check('L_final matches the formula', torch.allclose(expected, losses['loss']),
          '(%.6f vs %.6f)' % (expected.item(), losses['loss'].item()))

    # ------------------------------------------------------------ 4. backward
    print('\n== 4. backward ==')
    model.zero_grad()
    losses['loss'].backward()
    grad_norms = {
        'P_v_photo': model.prompt_photo.prompts.grad,
        'P_v_sketch': model.prompt_sketch.prompts.grad,
    }
    for d, p in model.prompt_text.ctx.items():
        grad_norms['P_t[%s]' % d] = p.grad
    for name, g in grad_norms.items():
        print('    %-16s grad_norm = %s' % (name, 'None' if g is None else '%.6e' % g.norm().item()))
        check('%s has a finite non-zero grad' % name,
              g is not None and torch.isfinite(g).all().item() and g.norm().item() > 0)
        # every prompted layer must be reached, not just layer 0
        if g is not None and g.shape[0] > 1:
            dead = [i for i in range(g.shape[0]) if g[i].norm().item() == 0]
            print('        per-layer: %s' % ['%.2e' % g[i].norm().item() for i in range(g.shape[0])])
            check('%s: every layer gets gradient' % name, not dead, 'dead layers %s' % dead)

    clip_with_grad = [n for n, p in model.clip.named_parameters() if p.grad is not None]
    check('no gradient reached the frozen CLIP', len(clip_with_grad) == 0,
          '' if not clip_with_grad else str(clip_with_grad[:3]))

    # ----------------------------------------------------------------- 5. GPA
    print('\n== 5. GPA (Eq. 6-7) ==')
    if model.gpa_photo is None:
        print('    GPA disabled (--gpa_enabled=0), skipped')
    else:
        w = model.gpa_photo.weights
        print('    gaussian weights (photo, first 5): %s ... sum=%.6f'
              % (['%.4f' % v for v in w[:5].tolist()], w.sum().item()))
        check('weights sum to 1', abs(w.sum().item() - 1.0) < 1e-5)
        check('photo and sketch aggregators are separate objects',
              model.gpa_photo is not model.gpa_sketch)

        with torch.no_grad():
            model.prompt_photo.prompts.fill_(1.0)
        model.gpa_photo.update(model.prompt_photo, 0)
        with torch.no_grad():
            model.prompt_photo.prompts.fill_(3.0)
        model.gpa_photo.update(model.prompt_photo, 1)

        w0, w1 = w[0].item(), w[1].item()
        want = (w0 * 1.0 + w1 * 3.0) / (w0 + w1)
        got = model.gpa_photo.aggregated()['prompts'].mean().item()
        print('    after epochs {0:P=1, 1:P=3}: aggregate=%.6f, expected=%.6f' % (got, want))
        check('GPA == gaussian weighted moving average', abs(got - want) < 1e-5)

        with torch.no_grad():
            model.prompt_photo.prompts.fill_(3.0)
        before = model.prompt_photo.prompts.clone()
        with use_aggregated_prompts(model._gpa_pairs()):
            inside = model.prompt_photo.prompts.mean().item()
        after = model.prompt_photo.prompts
        check('inference swaps in the GPA prompt', abs(inside - want) < 1e-5, '(%.6f)' % inside)
        check('training prompt is restored afterwards', torch.equal(before, after))

    # -------------------------------------------------------------- 6. depth
    print('\n== 6. deep prompts actually reach the deeper blocks ==')
    if opts.prompt_depth_vision > 1:
        with torch.no_grad():
            learner = model.prompt_photo
            shallow = model.clip.encode_image(img, learner.first(2), None)
            deep = model.clip.encode_image(img, learner.first(2), learner.deep())
        diff = (shallow - deep).abs().max().item()
        check('vision J_v>1 changes the output vs J_v=1', diff > 1e-4,
              '(J_v=%d, %d deep prompts, max abs diff %.6f)'
              % (opts.prompt_depth_vision, len(learner.deep()), diff))
    else:
        print('    prompt_depth_vision=1, nothing to check')

    if opts.prompt_depth_text > 1:
        pt = model.prompt_text
        tok = pt.tokenize(categories, model.device)
        with torch.no_grad():
            shallow_t = model.clip.encode_text(tok, pt.ctx_first('photo'), None)
            deep_t = model.clip.encode_text(tok, pt.ctx_first('photo'), pt.ctx_deep('photo'))
        diff_t = (shallow_t - deep_t).abs().max().item()
        check('text J_t>1 changes the output vs J_t=1', diff_t > 1e-4,
              '(J_t=%d, %d deep prompts, max abs diff %.6f)'
              % (opts.prompt_depth_text, len(pt.ctx_deep('photo')), diff_t))
    else:
        print('    prompt_depth_text=1, nothing to check')

    print('\n== summary ==')
    if failures:
        print('  %d CHECK(S) FAILED: %s' % (len(failures), failures))
        return 1
    print('  all checks passed')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
