from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange
import pdb

FLAGS = tf.app.flags.FLAGS
from ops import *
from utils import *

d_bn1 = batch_norm(name='d_bn1')
d_bn2 = batch_norm(name='d_bn2')
d_bn3 = batch_norm(name='d_bn3')

g_bn0 = batch_norm(name='g_bn0')
g_bn1 = batch_norm(name='g_bn1')
g_bn2 = batch_norm(name='g_bn2')
g_bn3 = batch_norm(name='g_bn3')

def select_g_d():
  return eval(FLAGS.G_type), eval(FLAGS.D_type)

def discriminator_dcgan(image, reuse=False, for_G=False):
  with tf.variable_scope('discriminator', reuse=reuse): 
  
    df_dim = 64

    h0 = lrelu(conv2d(image, df_dim, name='d_h0_conv'))
    h1 = lrelu(d_bn1(conv2d(h0, df_dim*2, name='d_h1_conv')))
    h2 = lrelu(d_bn2(conv2d(h1, df_dim*4, name='d_h2_conv')))
    h3 = lrelu(d_bn3(conv2d(h2, df_dim*8, name='d_h3_conv')))
    h4 = linear(tf.reshape(h3, [FLAGS.batch_size, -1]), 1, 'd_h4_logits')
    h4_sigmoid = tf.nn.sigmoid(h4, name='d_h4_sigmoid')

    return h4
def generator_dcgan(z, reuse=False, bn_train=True):
  with tf.variable_scope('generator', reuse=reuse):

    s = FLAGS.output_size
    s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)

    h0 = tf.nn.relu(g_bn0(linear(z, 64*8*s16*s16, 'g_h0_lin'), train=bn_train), name='g_h0_relu')
    h0 = tf.reshape(h0, [FLAGS.batch_size, s16, s16, 512])
    h1 = tf.nn.relu(g_bn1(deconv2d(h0,[FLAGS.batch_size, s8, s8, 256], name='g_h1'), train=bn_train),name='g_h1_relu')
    h2 = tf.nn.relu(g_bn2(deconv2d(h1,[FLAGS.batch_size, s4, s4, 128], name='g_h2'), train=bn_train),name='g_h2_relu')
    h3 = tf.nn.relu(g_bn3(deconv2d(h2,[FLAGS.batch_size, s2, s2, 64], name='g_h3'), train=bn_train),name='g_h3_relu')
    h4 = tf.nn.tanh(deconv2d(h3, [FLAGS.batch_size, s, s, FLAGS.c_dim], name='g_h4'), name='g_h4_sigmoid')

    return h4

def discriminator_no_bn(image, reuse=False, for_G=False):
  with tf.variable_scope('discriminator', reuse=reuse): 
  
    df_dim = 64

    h0 = lrelu(conv2d(image, df_dim, name='d_h0_conv'))
    h1 = lrelu((conv2d(h0, df_dim*2, name='d_h1_conv')))
    h2 = lrelu((conv2d(h1, df_dim*4, name='d_h2_conv')))
    h3 = lrelu((conv2d(h2, df_dim*8, name='d_h3_conv')))
    h4 = linear(tf.reshape(h3, [FLAGS.batch_size, -1]), 1, 'd_h4_logits')
    h4_sigmoid = tf.nn.sigmoid(h4, name='d_h4_sigmoid')

    return h4
def generator_no_bn(z, reuse=False, bn_train=True):
  with tf.variable_scope('generator', reuse=reuse):

    s = FLAGS.output_size
    s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)

    h0 = tf.nn.relu(linear(z, 64*8*s16*s16, 'g_h0_lin'), name='g_h0_relu')
    h0 = tf.reshape(h0, [FLAGS.batch_size, s16, s16, 512])
    h1 = tf.nn.relu(deconv2d(h0,[FLAGS.batch_size, s8, s8, 256], name='g_h1'),name='g_h1_relu')
    h2 = tf.nn.relu(deconv2d(h1,[FLAGS.batch_size, s4, s4, 128], name='g_h2'),name='g_h2_relu')
    h3 = tf.nn.relu(deconv2d(h2,[FLAGS.batch_size, s2, s2, 64], name='g_h3'),name='g_h3_relu')
    h4 = tf.nn.tanh(deconv2d(h3, [FLAGS.batch_size, s, s, FLAGS.c_dim], name='g_h4'), name='g_h4_sigmoid')

    return h4


def discriminator_mnist(image, reuse=False, for_G=False):
  with tf.variable_scope('discriminator', reuse=reuse): 

    h0 = lrelu(conv2d(image, 32, name='d_h0_conv'), name='d_h0_relu')
    h1 = lrelu(d_bn1(conv2d(h0, 64, name='d_h1_conv')), name='d_h1_relu')
    h1 = tf.reshape(h1, [FLAGS.batch_size, -1])
    h2 = lrelu(d_bn2(linear(h1, 1024, 'd_h2_lin')), name='d_h2_relu')
    h3 = linear(h2, 1, 'd_h3_lin')

    return h3

  
def generator_mnist(z, reuse=False, bn_train=True):
  with tf.variable_scope('generator', reuse=reuse):
    s2, s4 = int(FLAGS.output_size/2), int(FLAGS.output_size/4)
  
    h0 = tf.nn.relu(g_bn0(linear(z, 512, 'g_h0_lin'), train=bn_train), name='g_h0_relu')
    h1 = tf.nn.relu(g_bn1(linear(h0, 128*s4*s4, 'g_h1_lin'), train=bn_train), name='g_h1_relu')    
    h1 = tf.reshape(h1, [FLAGS.batch_size, s4, s4, 128])
    h2 = tf.nn.relu(g_bn2(deconv2d(h1, 
           [FLAGS.batch_size, s2, s2, 128], name='g_h2'), train=bn_train),name='g_h2_relu')
  
    h3 = tf.nn.sigmoid(deconv2d(h2, [FLAGS.batch_size, FLAGS.output_size, FLAGS.output_size, FLAGS.c_dim], name='g_h3'), name='g_h3_sigmoid')
    return h3


def inference(image, random_z):
  generator, discriminator = select_g_d()

  G_image = generator(random_z)

  D_logits_real = discriminator(image)

  D_logits_fake = discriminator(G_image, True)

  D_logits_fake_for_G = discriminator(G_image, True, True)

  return D_logits_real, D_logits_fake, D_logits_fake_for_G

def loss_l2(D_logits_real, D_logits_fake, D_logits_fake_for_G):
  G_loss = tf.reduce_mean(tf.nn.l2_loss(D_logits_fake_for_G -\
             tf.zeros_like(D_logits_fake_for_G))) 

  D_loss_real = tf.reduce_mean(tf.nn.l2_loss(D_logits_real - \
                  tf.ones_like(D_logits_real))) 

  D_loss_fake = tf.reduce_mean(tf.nn.l2_loss(D_logits_fake + \
                  tf.ones_like(D_logits_fake))) 

  D_loss = D_loss_real + D_loss_fake

  return G_loss, D_loss

def loss_sigmoid(D_logits_real, D_logits_fake, D_logits_fake_for_G):

  G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
             logits=D_logits_fake_for_G, labels=tf.ones_like(D_logits_fake_for_G)))

  D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                  logits=D_logits_real, labels=tf.ones_like(D_logits_real)))

  D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                  logits=D_logits_fake, labels=tf.zeros_like(D_logits_fake)))

  D_loss = D_loss_real + D_loss_fake

  return G_loss, D_loss


def train(G_loss, D_loss, G_vars, D_vars, global_step):

  G_optim = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=FLAGS.beta1) 
  D_optim = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=FLAGS.beta1) 

  G_grads = G_optim.compute_gradients(G_loss, var_list=G_vars)
  D_grads = D_optim.compute_gradients(D_loss, var_list=D_vars)

  G_train_op = G_optim.apply_gradients(G_grads, global_step=global_step)
  D_train_op = D_optim.apply_gradients(D_grads)

  return G_train_op, D_train_op
