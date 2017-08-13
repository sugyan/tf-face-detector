#!/bin/sh
# add Cloud SDK tools to your path

# export GCS_BUCKET=${YOUR_GCS_BUCKET}
if [ -z "$GCS_BUCKET" ]; then
    echo 'Set environment variable "GCS_BUCKET"'
    exit 1
fi

# data files
if gsutil -q stat gs://${GCS_BUCKET}/data/fddb_train.record; then
    echo "fddb_train.record exists."
else
    gsutil cp data/fddb_train.record gs://${YOUR_GCS_BUCKET}/data/fddb_train.record
fi
if gsutil -q stat gs://${GCS_BUCKET}/data/fddb_val.record; then
    echo "fddb_val.record exists."
else
    gsutil cp data/fddb_train.record gs://${YOUR_GCS_BUCKET}/data/fddb_val.record
fi
if gsutil -q stat gs://${GCS_BUCKET}/data/fddb_label_map.pbtxt; then
    echo "fddb_label_map.pbtxt exists."
else
    gsutil cp data/fddb_train.record gs://${YOUR_GCS_BUCKET}/data/fddb_label_map.pbtxt
fi

if gsutil -q stat gs://${GCS_BUCKET}/data/model.ckpt.data-00000-of-00001; then
    echo "model.ckpt.data-00000-of-00001 exists."
else
    if [ ! -f ssd_inception_v2_coco_11_06_2017.tar.gz ]; then
        wget http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_11_06_2017.tar.gz
    fi
    tar -xvf ssd_inception_v2_coco_11_06_2017.tar.gz
    gsutil cp ssd_inception_v2_coco_11_06_2017/model.ckpt.* gs://${GCS_BUCKET}/data/
fi

if gsutil -q stat gs://${GCS_BUCKET}/data/ssd_inception_v2_fddb.config; then
    echo "ssd_inception_v2_fddb.config exists."
else
    perl -i.bak -pe "s|PATH_TO_BE_CONFIGURED|"gs://${GCS_BUCKET}"/data|g" ./ssd_inception_v2_fddb.config
    gsutil cp ssd_inception_v2_fddb.config gs://${YOUR_GCS_BUCKET}/data/ssd_inception_v2_fddb.config
fi

# setup modules
if [ ! -f  models/dist/object_detection-0.1.tar.gz ]; then
    (cd models && python setup.py sdist)
fi
if [ ! -f  models/slim/dist/slim-0.1.tar.gz ]; then
    (cd models/slim && python setup.py sdist)
fi

# start training
gcloud ml-engine jobs submit training `whoami`_face_detection_`date +%s` \
    --job-dir=gs://${GCS_BUCKET}/train \
    --packages models/dist/object_detection-0.1.tar.gz,models/slim/dist/slim-0.1.tar.gz \
    --module-name object_detection.train \
    --region us-central1 \
    --config models/object_detection/samples/cloud/cloud.yml \
    -- \
    --train_dir=gs://${GCS_BUCKET}/train \
    --pipeline_config_path=gs://${GCS_BUCKET}/data/ssd_inception_v2_fddb.config
