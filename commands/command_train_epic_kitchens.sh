#!/bin/bash

PYTHONPATH=. python main.py --config-name=default_ssrl -m \
logging.name=epic_git2 \
data=epic_10s_audiopeak \
backbone/video=avid_r2plus1d_18 \
backbone/audio=avid_spec_cnn_9 \
optim.batch_size=128 \
environment.slurm=False \
environment.world_size=1 \
environment.multiprocessing_distributed=True \
environment.distributed=True \
environment.ngpu=4 \
environment.workers=32 \
environment.data_dir=/home/logs/ \
logging.save_freq=10 \
optim.epochs=100 \
optim.args.lr=0.005 \
criterion.args.clr_coeff=0.5 \
criterion.args.aot_coeff=0.5 \
data.args.base_path=/datasets/epic-kitchens/ \
data.args.delta_non_overlap=0.1 \
optim.use_lr_scheduler=True \
optim.lr_scheduler_args.max_lr=0.005 \
optim.lr_scheduler_args.total_steps=100 \
backbone.video.args.pretrained=True \
backbone.audio.args.pretrained=True
