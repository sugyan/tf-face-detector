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
    face_cascade = cv2.CascadeClassifier(os.path.join(cascades_dir, 'haarcascade_frontalface_default.xml'))
    eyes_cascade = cv2.CascadeClassifier(os.path.join(cascades_dir, 'haarcascade_eye.xml'))

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
                regions = []
                for _ in range(num):
                    regions.append(f.readline().strip())
                print(img_file, num, regions)
                img = cv2.imread(os.path.join(DIRECTORY, '{}.jpg'.format(img_file)))

                for region in regions:
                    e = region.split(' ')
                    size = max(float(e[0]), float(e[1])) * 1.1
                    # skip if face is too small
                    if size < 60.0:
                        break
                    center = (int(float(e[3]) + .5), int(float(e[4]) + .5))
                    angle = float(e[2]) / math.pi * 180.0
                    if angle < 0:
                        angle += 180.0
                    M = cv2.getRotationMatrix2D(center, angle - 90.0, 1)
                    M[0, 2] -= float(e[3]) - size
                    M[1, 2] -= float(e[4]) - size
                    # crop to detect frontalface
                    target = cv2.warpAffine(img, M, (int(size * 2 + .5), int(size * 2 + .5)))
                    faces = face_cascade.detectMultiScale(target)
                    if len(faces) != 1:
                        print('{} faces found...'.format(len(faces)))
                        break
                    rect = faces[0]
                    cv2.rectangle(target, tuple(rect[0:2]), tuple(rect[0:2] + rect[2:4]), (0, 255, 0))
                    face = target[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
                    eyes = []
                    for eye in eyes_cascade.detectMultiScale(face):
                        # reject false detection
                        if eye[1] > face.shape[0] / 2:
                            break
                        eyes.append(eye)
                    if len(eyes) != 2:
                        print('{} eyes found...'.format(len(faces)))
                        break
                    for eye in eyes:
                        cv2.rectangle(
                            target,
                            tuple(rect[0:2] + eye[0:2]),
                            tuple(rect[0:2] + eye[0:2] + eye[2:4]),
                            (0, 255, 255))
                    cv2.imshow('target', target)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
        break


if __name__ == '__main__':
    tf.app.run()
