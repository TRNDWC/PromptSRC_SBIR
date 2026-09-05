"""Training entry point for the PromptSRC-for-SBIR model.

Kept separate from ``experiments/LN_prompt.py`` on purpose: the original entry
point (frozen CLIP + t-SNE, trainer.validate only) is untouched, so rolling back
the prompt-learning work cannot break it.

Example:
    python -m experiments.promptsrc --exp_name=promptsrc \
        --data_dir=/path/to/Sketchy/ --max_epochs=20 --batch_size=64 \
        --prompt_depth_vision=9 --n_prompts_vision=4 \
        --prompt_depth_text=9   --n_prompts_text=4 \
        --n_text_templates=3
"""

import os

from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar

from src.model_promptsrc import PromptSRCModel
from src.dataset_retrieval import Sketchy, ValidDataset
from experiments.options import opts

if __name__ == '__main__':
    dataset_transforms = Sketchy.data_transform(opts)
    train_dataset = Sketchy(opts, dataset_transforms, mode='train', return_orig=False)
    train_loader = DataLoader(dataset=train_dataset, batch_size=opts.batch_size,
                              num_workers=opts.workers, shuffle=True, drop_last=True)

    val_sketch_loader = DataLoader(dataset=ValidDataset(opts, mode='sketch'),
                                   batch_size=opts.test_batch_size,
                                   num_workers=opts.workers, shuffle=False)
    val_photo_loader = DataLoader(dataset=ValidDataset(opts, mode='photo'),
                                  batch_size=opts.test_batch_size,
                                  num_workers=opts.workers, shuffle=False)

    logger = TensorBoardLogger('tb_logs', name=opts.exp_name)
    checkpoint_callback = ModelCheckpoint(
        monitor='mAP',
        dirpath='saved_models/%s' % opts.exp_name,
        filename='{epoch:02d}-{mAP:.4f}',
        mode='max',
        save_last=True)

    # Lightning 2.x picks RichProgressBar automatically when `rich` is
    # installed, and Rich's live display needs a real TTY -- through a pipe,
    # tee or nohup it reprints a whole line per refresh instead of overwriting.
    # tqdm degrades gracefully, and refresh_rate keeps the volume sane either
    # way. --progress_refresh=0 turns the bar off for good (log files, cron).
    callbacks = [checkpoint_callback]
    if opts.progress_refresh > 0:
        callbacks.append(TQDMProgressBar(refresh_rate=opts.progress_refresh))

    ckpt_path = os.path.join('saved_models', opts.exp_name, 'last.ckpt')
    if not os.path.exists(ckpt_path):
        ckpt_path = None
    else:
        print('resuming training from %s' % ckpt_path)

    trainer = Trainer(accelerator='gpu', devices=1,
                      min_epochs=1, max_epochs=opts.max_epochs,
                      benchmark=True,
                      logger=logger,
                      check_val_every_n_epoch=1,
                      enable_progress_bar=opts.progress_refresh > 0,
                      callbacks=callbacks)

    model = PromptSRCModel()
    trainer.fit(model,
                train_dataloaders=train_loader,
                val_dataloaders=[val_sketch_loader, val_photo_loader],
                ckpt_path=ckpt_path)
