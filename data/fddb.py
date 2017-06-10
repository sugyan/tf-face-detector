import cv2
import math
import os
import tarfile

import tensorflow as tf

IMAGES_DOWNLOAD_URL = 'http://tamaraberg.com/faceDataset/originalPics.tar.gz'
ANNOTATIONS_DOWNLOAD_URL = 'http://vis-www.cs.umass.edu/fddb/FDDB-folds.tgz'

DIRECTORY = os.path.join(os.path.dirname(__file__), 'fddb')


def download_and_extract():
    download = tf.contrib.learn.datasets.base.maybe_download
    for target in [IMAGES_DOWNLOAD_URL, ANNOTATIONS_DOWNLOAD_URL]:
        filename = os.path.basename(target)
        filepath = download(filename, DIRECTORY, target)
        with tarfile.open(filepath) as tar:
            for file in tar:
                path = os.path.join(DIRECTORY, file.name)
                if not os.path.exists(path):
                    print('extract {}'.format(path))
                    tar.extract(file, path=DIRECTORY)


def main(argv=None):
    # download_and_extract()
    cascades_dir = os.path.normpath(os.path.join(cv2.__file__, '..', '..', '..', '..', 'share', 'OpenCV', 'haarcascades'))
    cascade = cv2.CascadeClassifier(os.path.join(cascades_dir, 'haarcascade_frontalface_default.xml'))

    folds_dir = os.path.join(DIRECTORY, 'FDDB-folds')
    for filename in os.listdir(folds_dir):
        if not filename.endswith('-ellipseList.txt'):
            continue
        with open(os.path.join(folds_dir, filename)) as f:
            for line in f:
                img_file = line.strip()
                if len(img_file) == 0:
                    break
                num = int(f.readline())
                faces = []
                for _ in range(num):
                    faces.append(f.readline().strip())
                print(img_file, num, faces)
                img = cv2.imread(os.path.join(DIRECTORY, '{}.jpg'.format(img_file)))
                detected = cascade.detectMultiScale(img)

                for face in detected:
                    hw = int((face[2] + 1.0) / 2.0)
                    hh = int((face[3] + 1.0) / 2.0)
                    cv2.ellipse(
                        img,
                        (face[0] + hw, face[1] + hh),
                        (hw, hh),
                        0,
                        0, 360,
                        (255, 255, 0))
                for face in faces:
                    e = face.split(' ')
                    cv2.ellipse(
                        img,
                        (int(float(e[3])), int(float(e[4]))),
                        (int(float(e[0])), int(float(e[1]))),
                        float(e[2]) / math.pi * 180.0,
                        0, 360,
                        (0, 255, 0))
                cv2.imshow('image', img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        break


if __name__ == '__main__':
    tf.app.run()
