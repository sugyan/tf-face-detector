import os
import tarfile

import tensorflow as tf

IMAGES_DOWNLOAD_URL = 'http://tamaraberg.com/faceDataset/originalPics.tar.gz'
ANNOTATIONS_DOWNLOAD_URL = 'http://vis-www.cs.umass.edu/fddb/FDDB-folds.tgz'


def main(argv=None):
    download = tf.contrib.learn.datasets.base.maybe_download

    directory = os.path.join(os.path.dirname(__file__), 'fddb')
    for target in [IMAGES_DOWNLOAD_URL, ANNOTATIONS_DOWNLOAD_URL]:
        filename = os.path.basename(target)
        filepath = download(filename, directory, target)
        with tarfile.open(filepath) as tar:
            for file in tar:
                path = os.path.join(directory, file.name)
                if not os.path.exists(path):
                    print('extract {}'.format(path))
                    tar.extract(file, path=directory)


if __name__ == '__main__':
    tf.app.run()
