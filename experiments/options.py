import argparse

parser = argparse.ArgumentParser(description='Sketch-based OD')

parser.add_argument('--exp_name', type=str, default='LN_prompt')

# --------------------
# DataLoader Options
# --------------------

# Path to 'Sketchy' folder holding Sketch_extended dataset. It should have 2 folders named 'sketch' and 'photo'.
parser.add_argument('--data_dir', type=str, default='/isize2/sain/data/Sketchy/') 
parser.add_argument('--max_size', type=int, default=224)
parser.add_argument('--nclass', type=int, default=10)
parser.add_argument('--data_split', type=float, default=-1.0)

# ----------------------
# Training Params
# ----------------------

parser.add_argument('--clip_lr', type=float, default=1e-4)
parser.add_argument('--clip_LN_lr', type=float, default=1e-6)
parser.add_argument('--prompt_lr', type=float, default=1e-4)
parser.add_argument('--linear_lr', type=float, default=1e-4)
parser.add_argument('--batch_size', type=int, default=192)
parser.add_argument('--test_batch_size', type=int, default=1024)
parser.add_argument('--workers', type=int, default=4)

# ----------------------
# ViT Prompt Parameters
# ----------------------
parser.add_argument('--prompt_dim', type=int, default=768)
parser.add_argument('--n_prompts', type=int, default=3)

# ------------------------------------------------------------------
# PromptSRC-for-SBIR (src/model_promptsrc.py)
# ------------------------------------------------------------------
parser.add_argument('--max_epochs', type=int, default=20,
                    help='GPA weights are defined over this horizon (Eq. 6-7)')
parser.add_argument('--train_layernorm', type=int, default=1,
                    help='unfreeze every LayerNorm of both CLIP encoders (lr = --clip_LN_lr)')
parser.add_argument('--anchor_uses_original_ln', type=int, default=1,
                    help='SCL anchors run with the pre-trained LayerNorm weights, so the '
                         'regularisation target does not drift with the model')
parser.add_argument('--clip_float', type=int, default=1,
                    help='cast the frozen CLIP to fp32; fp16 underflows the KL/L1 SCL losses')

# prompt geometry: J = depth, V/T = number of tokens
parser.add_argument('--prompt_depth_vision', type=int, default=9)
parser.add_argument('--n_prompts_vision', type=int, default=4)
parser.add_argument('--prompt_depth_text', type=int, default=9)
parser.add_argument('--n_prompts_text', type=int, default=4)
parser.add_argument('--prompt_init_std', type=float, default=0.02)

# text branch
parser.add_argument('--text_prompt_mode', type=str, default='shared',
                    choices=['shared', 'per_domain'],
                    help='one shared P_t, or separate P_t per domain')
parser.add_argument('--text_ctx_init_photo', type=str, default='a photo of a')
parser.add_argument('--text_ctx_init_sketch', type=str, default='a sketch of a')
parser.add_argument('--n_text_templates', type=int, default=3,
                    help='N handcrafted templates averaged into the frozen text anchor. '
                         'PLACEHOLDER: the paper uses N=60 and its Fig. 5 ablation shows '
                         'larger N helps; N=3 is only meant to get a run going.')
parser.add_argument('--text_scl_domains', type=str, default='photo,sketch',
                    help='domains averaged in L_SCL_text')

# loss weights -- giá trị tạm từ PromptSRC, chưa tune cho SBIR
parser.add_argument('--lambda_scl_image', type=float, default=10.0)   # lambda1
parser.add_argument('--lambda_scl_text', type=float, default=25.0)    # lambda2
parser.add_argument('--lambda_scl_logits', type=float, default=1.0)
parser.add_argument('--scl_normalize_features', type=int, default=1,
                    help='L2-normalise both sides of the SCL feature L1 losses')
parser.add_argument('--scl_logits_anchor', type=str, default='sketch',
                    choices=['sketch', 'photo'],
                    help='which image branch feeds the SCL-logits pair (design hypothesis, see model)')

# GPA -- per branch mu / sigma^2
parser.add_argument('--gpa_enabled', type=int, default=1)
parser.add_argument('--gpa_apply_text', type=int, default=1)
parser.add_argument('--gpa_mu_photo', type=float, default=10.0)
parser.add_argument('--gpa_sigma2_photo', type=float, default=30.0)
parser.add_argument('--gpa_mu_sketch', type=float, default=10.0)
parser.add_argument('--gpa_sigma2_sketch', type=float, default=30.0)
parser.add_argument('--gpa_mu_text', type=float, default=10.0)
parser.add_argument('--gpa_sigma2_text', type=float, default=30.0)

# ``parse_known_args`` (not ``parse_args``) so that importing this module from a
# test/sanity script with its own CLI flags does not blow up.
opts, _unknown_opts = parser.parse_known_args()
