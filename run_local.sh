#!/bin/sh

# setup trained data
if [ ! -f data/ssd_inception_v2_coco_11_06_2017.tar.gz ]; then
    wget -P data http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_11_06_2017.tar.gz
fi
if [ ! -d data/ssd_inception_v2_coco_11_06_2017 ]; then
    tar -xvf data/ssd_inception_v2_coco_11_06_2017.tar.gz -C data
fi
ln -s $(pwd)/data/ssd_inception_v2_coco_11_06_2017/model.ckpt.* data

perl -pe "s|PATH_TO_BE_CONFIGURED|${PWD}/data|g" ./ssd_inception_v2_fddb.config.base > ssd_inception_v2_fddb.config

# start training
(cd models && protoc object_detection/protos/*.proto --python_out=.)
export PYTHONPATH=$PYTHONPATH:`pwd`/models:`pwd`/models/slim
python models/object_detection/train.py \
    --logtostderr \
    --pipeline_config_path=./ssd_inception_v2_fddb.config \
    --train_dir=../train
