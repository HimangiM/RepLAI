PRETRAINED_DIR=$1
shift 1

EGO4D_ANNOTS=$PWD/data/long_term_anticipation/annotations/
EGO4D_VIDEOS=$PWD/data/long_term_anticipation/clips/

WORK_DIR=$PRETRAINED_DIR/eval_ego4d_lta
mkdir -p ${WORK_DIR}
echo Working directory ${WORK_DIR}

# convert pretrained weights to ego4d format
PRETRAINED_WEIGHTS=$PRETRAINED_DIR/checkpoints/checkpoint_latest.pth
python tools/long_term_anticipation/convert_checkpoint.py $PRETRAINED_WEIGHTS
PRETRAINED_WEIGHTS=$PRETRAINED_DIR/checkpoints/checkpoint_latest-ego.pth

# train forecasting model
python -m scripts.run_lta \
  --working_directory ${WORK_DIR} \
  --cfg configs/Ego4dLTA/R2PLUS1D_8x8_R101.yaml \
  DATA.PATH_TO_DATA_DIR ${EGO4D_ANNOTS} \
  DATA.PATH_PREFIX ${EGO4D_VIDEOS} \
  CHECKPOINT_LOAD_MODEL_HEAD False \
  MODEL.FREEZE_BACKBONE True \
  DATA.CHECKPOINT_MODULE_FILE_PATH $PRETRAINED_WEIGHTS \
  FORECASTING.AGGREGATOR TransformerAggregator \
  FORECASTING.DECODER MultiHeadDecoder \
  FORECASTING.NUM_INPUT_CLIPS 4 \
  SOLVER.BASE_LR 0.01 \
  TRAIN.BATCH_SIZE 32 \
  TEST.BATCH_SIZE 32 \
  NUM_GPUS 8 \
  $@
