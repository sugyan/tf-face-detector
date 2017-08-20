# tf-face-detector

Face Detector using [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/object_detection)


## Prerequisite

- Python >= 3.x
  - TensorFlow >= 1.2
  - Pillow >= 4.2.1 (for visualizing results)
  - cv2 >= 3.3 (for generating dataset)


## Setup

```
git submodule update --init
pip3 install -r requirements.txt
```


## FDDB dataset

http://vis-www.cs.umass.edu/fddb/

To download data and generate tfrecord dataset (needed `cv2`):

```
python data/fddb.py
```


## Training

```sh
perl -pe "s|PATH_TO_BE_CONFIGURED|${PWD}/data|g" ./ssd_inception_v2_fddb.config.base > ssd_inception_v2_fddb.config
(cd models && protoc object_detection/protos/*.proto --python_out=.)
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
python object_detection/train.py \
    --train_dir=../train \
    --pipeline_config_path=../ssd_inception_v2_faces.config
```

or

```sh
./run_local.sh
```

## Export graph

```sh
export PYTHONPATH=${PYTHONPATH}:$(pwd)/models:$(pwd)/models/slim
export CHECKPOINT_NUMBER=<target checkpoint number>
export EXPORT_DIRECTORY=<path to output graph>
python models/object_detection/export_inference_graph.py \
    --input_type=encoded_image_string_tensor \
    --pipeline_config_path=ssd_inception_v2_fddb.config \
    --trained_checkpoint_prefix=train/model.ckpt-${CHECKPOINT_NUMBER} \
    --output_directory=${EXPORT_DIRECTORY}
```
