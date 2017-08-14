import os
import numpy as np
import tensorflow as tf
from models.object_detection.utils.visualization_utils import visualize_boxes_and_labels_on_image_array
from PIL import Image

flags = tf.app.flags
flags.DEFINE_string('model_directory', None,
                    'Path to directory where exported graph is saved.')
flags.DEFINE_string('images_directory', os.path.join(os.path.dirname(__file__), '..', 'images'),
                    'Path to target images directory.')

FLAGS = flags.FLAGS


def main(argv=None):
    assert FLAGS.model_directory, '`model_directory` is missing'
    # import graph
    graph_def = tf.GraphDef()
    with tf.gfile.GFile(os.path.join(FLAGS.model_directory, 'frozen_inference_graph.pb'), 'rb') as f:
        graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

    with tf.Session() as sess:
        encoded_image = sess.graph.get_tensor_by_name('encoded_image_string_tensor:0')
        fetches = {
            'image': tf.image.decode_jpeg(tf.squeeze(encoded_image)),
            'boxes': sess.graph.get_tensor_by_name('detection_boxes:0'),
            'scores': sess.graph.get_tensor_by_name('detection_scores:0'),
            'classes': sess.graph.get_tensor_by_name('detection_classes:0'),
            'num_detections': sess.graph.get_tensor_by_name('num_detections:0'),
        }
        # read image files, detect faces
        for file in os.listdir(FLAGS.images_directory):
            if not file.endswith('.jpg'):
                continue
            filepath = os.path.join(FLAGS.images_directory, file)
            with tf.gfile.GFile(filepath, 'rb') as f:
                jpeg = f.read()
            results = sess.run(fetches, feed_dict={encoded_image: [jpeg]})

            category_index = {
                1: {'name': 'face'},
                2: {'name': 'eye'}
            }
            image = results['image']
            visualize_boxes_and_labels_on_image_array(
                image,
                np.squeeze(results['boxes']),
                np.squeeze(results['classes']).astype(np.int32),
                np.squeeze(results['scores']),
                category_index,
                use_normalized_coordinates=True)
            Image.fromarray(image).show()


if __name__ == '__main__':
    tf.app.run()
