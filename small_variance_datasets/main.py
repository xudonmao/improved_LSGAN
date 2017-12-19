from __future__ import absolute_import
from __future__ import division
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import pdb
import arch 
import os
import data

from utils import *


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_float('learning_rate', '0.0002', """learning rate""")
tf.app.flags.DEFINE_float('beta1', '0.5', """beta for Adam""")
tf.app.flags.DEFINE_integer('batch_size', '64', """batch size""")
tf.app.flags.DEFINE_integer('z_dim', '100', """z dimsion""")
tf.app.flags.DEFINE_integer('c_dim', '1', """c dimsion""")
tf.app.flags.DEFINE_integer('output_size', '28', """output size""")
tf.app.flags.DEFINE_integer('max_steps', 50001, """Number of batches to run.""")
tf.app.flags.DEFINE_string('G_type', 'generator_digit', """Directory where to write event logs """)
tf.app.flags.DEFINE_string('D_type', 'discriminator_digit', """Directory where to write event logs """)
tf.app.flags.DEFINE_string('loss', 'lsgan', """Loss type""")
tf.app.flags.DEFINE_string('log_dir', './log/', """Directory where to write event logs """)
tf.app.flags.DEFINE_string('checkpoint_dir', './checkpoint/', """Directory where to write the checkpoint""")
tf.app.flags.DEFINE_string('data_path', './times28_shift_rotate.tfrecords', """Path to the lsun data file""")
tf.app.flags.DEFINE_string('data_dir', './data/', """data dir""")


def sess_init():
  init = tf.global_variables_initializer()
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  sess = tf.Session(config=config)
  sess.run(init)
  return sess
def GetVars():
  t_vars = tf.trainable_variables()
  G_vars = [var for var in t_vars if 'g_' in var.name]
  D_vars = [var for var in t_vars if 'd_' in var.name]
  return G_vars, D_vars

def Holder():
  z_h = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, 100])
  images_h = tf.placeholder(tf.float32, 
      shape=[FLAGS.batch_size, FLAGS.output_size, FLAGS.output_size, FLAGS.c_dim])
  return z_h, images_h

def GenValsForHolder(data_set, sess):
  z_v = np.random.uniform(-1, 1, 
	  [FLAGS.batch_size, FLAGS.z_dim]).astype(np.float32)
  images_v = sess.run(data_set)
  #images_v, _ = data_set.next_batch()

  return z_v, images_v

def train():
  with tf.Graph().as_default():


    global_step = tf.Variable(0, trainable=False)

    z_h, images_h = Holder()

    D_logits_real, D_logits_fake, D_logits_fake_for_G = \
      arch.inference(images_h, z_h)

    sampler = eval('arch.'+FLAGS.G_type)(z_h, reuse = True, bn_train = False)

    if FLAGS.loss == 'lsgan':
      G_loss, D_loss = arch.loss_l2(D_logits_real, D_logits_fake, D_logits_fake_for_G)
    else:
      G_loss, D_loss = arch.loss_sigmoid(D_logits_real, D_logits_fake, D_logits_fake_for_G)

    G_vars, D_vars = GetVars()

    G_train_op, D_train_op = arch.train(G_loss, D_loss, G_vars, D_vars, global_step)
    
    data_set = data.DATA(FLAGS.batch_size).load()

    sess = sess_init()

    tf.train.start_queue_runners(sess=sess)

    saver = tf.train.Saver()

    for step in xrange(0, FLAGS.max_steps):
      z_v, images_v = GenValsForHolder(data_set, sess)

      _, errD = sess.run([D_train_op, D_loss],
          feed_dict={ z_h: z_v, images_h:images_v})

      _, errG = sess.run([G_train_op, G_loss],
          feed_dict={ z_h: z_v})

      if step % 100 == 0:
        print "step = %d, errD = %f, errG = %f" % (step, errD, errG)


      if step % 1000 == 0:
        samples = sess.run(sampler, 
          feed_dict={z_h: z_v})
        save_images(samples, [8, 8],
            './samples/train_{:d}.png'.format(step))
        if step<=10000:
          save_images(images_v, [8, 8],
            './samples_real/train_{:d}.png'.format(step))
                            
      if step % 10000 == 0:
        saver.save(sess, '{0}/{1}.model'.format(FLAGS.checkpoint_dir, step), global_step)

def main(argv=None):
  os.system('mkdir -p samples samples_real')
  os.system('mkdir -p {0}'.format(FLAGS.checkpoint_dir))
  train()

if __name__ == "__main__":
  tf.app.run()
