#!/bin/bash

PYTHONPATH=. python main.py --config-name=default_lincls -m \
logging.name=log_train_epickitchens data=epic_action_segments \
backbone/video=avid_r2plus1d_18 environment.slurm=False \
environment.world_size=1 environment.multiprocessing_distributed=True \
environment.distributed=True environment.ngpu=4 environment.workers=8 \
environment.data_dir=/home/data/ \
logging.save_freq=10 optim=sgd_lincls optim.epochs=30 optim.schedule=[18,24] optim.batch_size=128 \
optim.args.lr=0.005 environment.resume=False \
logging.suffix="_eval" train_algo.normalize_feats=False