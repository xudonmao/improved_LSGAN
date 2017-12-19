from __future__ import absolute_import
from __future__ import division
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import os
import pdb

output_size = 64
data_path = './bedroom64.tfrecords'

class DATA():
  def __init__(self, batch_size):
    self._batch_size = batch_size

  def read_and_decode(self, filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
        })
  
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image.set_shape([output_size*output_size*3])
    image = tf.reshape(image, [output_size,output_size,3])
  
    image = tf.cast(image, tf.float32) 
    #image = tf.cast(image, tf.float32) * (1. / 127.5) - 1.0
  
    return image
  
  
  def load(self):
  
    filename_queue = tf.train.string_input_producer(
        [data_path])
  
    image = self.read_and_decode(filename_queue)
  
    images = tf.train.shuffle_batch(
        [image], batch_size=self._batch_size, num_threads=2,
        capacity=1000 + 3 * self._batch_size,
        min_after_dequeue=1000)
  
    return images

