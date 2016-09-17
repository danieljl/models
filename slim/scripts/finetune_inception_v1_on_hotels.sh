#!/bin/bash
#
# This script performs the following operations:
# 1. Fine-tunes an InceptionV1 model on the Hotels training set.
# 2. Evaluates the model on the Hotels validation set.
#
# Usage:
# cd slim
# ./slim/scripts/finetune_inception_v1_on_hotels.sh

# Where the pre-trained InceptionV1 checkpoint is saved to.
PRETRAINED_CHECKPOINT_DIR=/data/training/checkpoints

# Where the training (fine-tuned) checkpoint and logs will be saved to.
TRAIN_DIR=/data/training/inception_v1

# Where the dataset is saved to.
DATASET_DIR=/data/dataset-all

THIS_DIR=$(dirname $(readlink -f "$0"))


echo '------------------- DOWNLOAD PRE-TRAINED CHECKPOINT -------------------'
# Download the pre-trained checkpoint.
if [ ! -d "$PRETRAINED_CHECKPOINT_DIR" ]; then
  mkdir ${PRETRAINED_CHECKPOINT_DIR}
fi
if [ ! -f ${PRETRAINED_CHECKPOINT_DIR}/inception_v1.ckpt ]; then
  pushd /tmp
  wget http://download.tensorflow.org/models/inception_v1_2016_08_28.tar.gz
  tar -xvf inception_v1_2016_08_28.tar.gz
  mv inception_v1.ckpt ${PRETRAINED_CHECKPOINT_DIR}/inception_v1.ckpt
  rm inception_v1_2016_08_28.tar.gz
  popd
fi
echo
echo '======================================================================='
echo -ne '\n\n\n\n\n\n\n\n'

# echo '------------------------ TRAIN THE LAST LAYER -------------------------'
# # Fine-tune only the new layers.
# python "$THIS_DIR/../train_image_classifier.py" \
#   --train_dir=${TRAIN_DIR} \
#   --dataset_name=hotels \
#   --dataset_split_name=train \
#   --dataset_dir=${DATASET_DIR} \
#   --model_name=inception_v1 \
#   --checkpoint_path=${PRETRAINED_CHECKPOINT_DIR}/inception_v1.ckpt \
#   --checkpoint_exclude_scopes=InceptionV1/Logits \
#   --trainable_scopes=InceptionV1/Logits \
#   --max_number_of_steps=300000 \
#   --batch_size=32 \
#   --learning_rate=0.01 \
#   --save_interval_secs=3600 \
#   --save_summaries_secs=120 \
#   --log_every_n_steps=100 \
#   --optimizer=rmsprop \
#   --weight_decay=0.00004 \
#   --num_clones=4 \
#   --num_readers=32 \
#   --num_preprocessing_threads=32
# echo
# echo '======================================================================='
# echo -ne '\n\n\n\n\n\n\n\n'

# echo '----------------- EVALUATE TRAINING OF THE LAST LAYER -----------------'
# # Run evaluation.
# python "$THIS_DIR/../eval_image_classifier.py" \
#   --checkpoint_path=${TRAIN_DIR} \
#   --eval_dir=${TRAIN_DIR} \
#   --dataset_name=hotels \
#   --dataset_split_name=validation \
#   --dataset_dir=${DATASET_DIR} \
#   --model_name=inception_v1 \
#   --num_preprocessing_threads=32
# echo
# echo '======================================================================='
# echo -ne '\n\n\n\n\n\n\n\n'

echo '-------------------------- TRAIN ALL LAYERS ---------------------------'
# Fine-tune all the new layers.
python "$THIS_DIR/../train_image_classifier.py" \
  --train_dir=${TRAIN_DIR}/all \
  --dataset_name=hotels \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --checkpoint_path=${TRAIN_DIR} \
  --model_name=inception_v1 \
  --max_number_of_steps=100000 \
  --batch_size=32 \
  --learning_rate=0.001 \
  --save_interval_secs=3600 \
  --save_summaries_secs=120 \
  --log_every_n_steps=100 \
  --optimizer=rmsprop \
  --weight_decay=0.00004 \
  --num_clones=4 \
  --num_readers=32 \
  --num_preprocessing_threads=32
echo
echo '======================================================================='
echo -ne '\n\n\n\n\n\n\n\n'

echo '------------------- EVALUATE TRAINING OF ALL LAYERS -------------------'
# Run evaluation.
python "$THIS_DIR/../eval_image_classifier.py" \
  --checkpoint_path=${TRAIN_DIR}/all \
  --eval_dir=${TRAIN_DIR}/all \
  --dataset_name=hotels \
  --dataset_split_name=validation \
  --dataset_dir=${DATASET_DIR} \
  --model_name=inception_v1 \
  --num_preprocessing_threads=32
echo
echo '======================================================================='
echo -ne '\n\n\n\n\n\n\n\n'
