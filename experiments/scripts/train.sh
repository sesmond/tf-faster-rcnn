#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

# 训练数据
Date=$(date +%Y%m%d%H%M)


echo "选择您的操作"
select var in "train" "console" "stop"; do
  break;
done
echo "You have selected $var "


if [ "$var" = "stop" ]; then
    echo "停止训练"
    ps aux|grep python|grep name=faster-rcnn|awk '{print $2}'|xargs kill -9
    exit
fi


#echo "输入训练要用的GPU"
read -p "输入训练要用的GPU:" GPU_ID ;
echo "您选择了GPU $GPU_ID 进行训练"


echo "选择您要使用的网络"
select NET in "res101" "vgg16"; do
  break;
done
echo "You have selected $NET "

#TODO 数据集选择，多选等
DATASET=gen


array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

# TODO 数据集指定
TRAIN_IMDB="gen_train"
TEST_IMDB="gen_test"
STEPSIZE="[50000]"
ITERS=70000
ANCHORS="[8,16,32]"
RATIOS="[0.5,1,2]"


LOG="experiments/logs/${NET}_${TRAIN_IMDB}_${EXTRA_ARGS_SLUG}_${NET}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

set +x
if [[ ! -z  ${EXTRA_ARGS_SLUG}  ]]; then
  NET_FINAL=output/${NET}/${TRAIN_IMDB}/${EXTRA_ARGS_SLUG}/${NET}_faster_rcnn_iter_${ITERS}.ckpt
else
  NET_FINAL=output/${NET}/${TRAIN_IMDB}/default/${NET}_faster_rcnn_iter_${ITERS}.ckpt
fi
set -x

if [ ! -f ${NET_FINAL}.index ]; then
  if [[ ! -z  ${EXTRA_ARGS_SLUG}  ]]; then
    CUDA_VISIBLE_DEVICES=${GPU_ID}
    nohup  time python ./tools/trainval_net.py \
      --weight data/imagenet_weights/${NET}.ckpt \
      --imdb ${TRAIN_IMDB} \
      --imdbval ${TEST_IMDB} \
      --iters ${ITERS} \
      --cfg experiments/cfgs/${NET}.yml \
      --tag ${EXTRA_ARGS_SLUG} \
      --net ${NET} \
      --set ANCHOR_SCALES ${ANCHORS} ANCHOR_RATIOS ${RATIOS} \
      TRAIN.STEPSIZE ${STEPSIZE} ${EXTRA_ARGS} \
      >> ./logs/console_$Date.log 2>&1
  else
    CUDA_VISIBLE_DEVICES=${GPU_ID}
    nohup time python ./tools/trainval_net.py \
      --weight data/imagenet_weights/${NET}.ckpt \
      --imdb ${TRAIN_IMDB} \
      --imdbval ${TEST_IMDB} \
      --iters ${ITERS} \
      --cfg experiments/cfgs/${NET}.yml \
      --net ${NET} \
      --set ANCHOR_SCALES ${ANCHORS} ANCHOR_RATIOS ${RATIOS} \
      TRAIN.STEPSIZE ${STEPSIZE} ${EXTRA_ARGS} \
      >> ./logs/console_$Date.log 2>&1

  fi
fi

#./experiments/scripts/test_faster_rcnn.sh $@
