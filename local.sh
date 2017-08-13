#!/bin/sh

# setup trained data
if [ ! -f data/ssd_inception_v2_coco_11_06_2017.tar.gz ]; then
    wget -P data http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_11_06_2017.tar.gz
fi
tar -xvf data/ssd_inception_v2_coco_11_06_2017.tar.gz -C data
ln -s $(pwd)/data/ssd_inception_v2_coco_11_06_2017/model.ckpt.* data

perl -i.bak -pe "s|PATH_TO_BE_CONFIGURED|${PWD}/data|g" ./ssd_inception_v2_fddb.config

# start training
cd models
protoc object_detection/protos/*.proto --python_out=.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
python object_detection/train.py \
    --logtostderr \
    --pipeline_config_path=../ssd_inception_v2_fddb.config \
    --train_dir=../train
