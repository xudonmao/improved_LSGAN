from __future__ import absolute_import
from __future__ import division

import os
import pdb
import tensorflow as tf
import numpy as np
import random
import scipy.misc
import sys



tf.app.flags.DEFINE_string('directory', './',
                           'Directory to download data files and write the '
                           'converted result')
tf.app.flags.DEFINE_integer('validation_size', 5000,
                            'Number of examples to separate from the training '
                            'data for the validation set.')
FLAGS = tf.app.flags.FLAGS

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def img_to_tf(file_list, name):
  #pdb.set_trace()
  filename = os.path.join(FLAGS.directory, name + '.tfrecords')
  writer = tf.python_io.TFRecordWriter(filename)
  
  with open(file_list) as f:
    file_set = [line.strip('\n') for line in f]
  random.shuffle(file_set)

  output_size = 128

  for i, line in enumerate(file_set):
    img_path = line
    img = scipy.misc.imread(img_path)
    img = scipy.misc.imresize(img, [output_size, output_size])

    rows = img.shape[0]
    cols = img.shape[1]
    depth = 3

    image_raw = img.tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'height': _int64_feature(rows),
        'width': _int64_feature(cols),
        'depth': _int64_feature(depth),
        'image_raw': _bytes_feature(image_raw)}))
    writer.write(example.SerializeToString())
    if i % 10000 == 0:
      print i

  writer.close()



def main(argv):
  img_to_tf(sys.argv[1], sys.argv[2])

if __name__ == '__main__':
  tf.app.run()
