# tf-face-detector

## FDDB

http://vis-www.cs.umass.edu/fddb/

```
$ python data/fddb.py
```


## Training


```
### Download COCO-pretrained SSD with Inception V2 model
$ cd ./data
$ wget http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_11_06_2017.tar.gz
$ tar -xvf ssd_inception_v2_coco_11_06_2017.tar.gz

### Training
$ cd ../models
$ protoc object_detection/protos/*.proto --python_out=.
$ export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
$ python object_detection/train.py \
      --train_dir=../train \
      --pipeline_config_path=../ssd_inception_v2_faces.config
```
