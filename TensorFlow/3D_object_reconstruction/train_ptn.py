# Copyright 2017 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Contains training plan for the Im2vox model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf

from tensorflow import app

import model_ptn

flags = tf.app.flags
slim = tf.contrib.slim

flags.DEFINE_string('inp_dir',
                    '',
                    'Directory path containing the input data (tfrecords).')
flags.DEFINE_string(
    'dataset_name', 'shapenet_chair',
    'Dataset name that is to be used for training and evaluation.')
flags.DEFINE_integer('z_dim', 512, '')
flags.DEFINE_integer('f_dim', 64, '')
flags.DEFINE_integer('fc_dim', 1024, '')
flags.DEFINE_integer('num_views', 24, 'Num of viewpoints in the input data.')
flags.DEFINE_integer('image_size', 64,
                     'Input images dimension (pixels) - width & height.')
flags.DEFINE_integer('vox_size', 32, 'Voxel prediction dimension.')
flags.DEFINE_integer('step_size', 24, 'Steps to take in rotation to fetch viewpoints.')
flags.DEFINE_integer('batch_size', 6, 'Batch size while training.')
flags.DEFINE_float('focal_length', 0.866, 'Focal length parameter used in perspective projection.')
flags.DEFINE_float('focal_range', 1.732, 'Focal length parameter used in perspective projection.')
flags.DEFINE_string('encoder_name', 'ptn_encoder',
                    'Name of the encoder network being used.')
flags.DEFINE_string('decoder_name', 'ptn_vox_decoder',
                    'Name of the decoder network being used.')
flags.DEFINE_string('projector_name', 'perspective_projector',
                    'Name of the projector network being used.')
# Save options
flags.DEFINE_string('checkpoint_dir', '/tmp/ptn_train/',
                    'Directory path for saving trained models and other data.')
flags.DEFINE_string('model_name', 'ptn_finetune',
                    'Name of the model used in naming the TF job. Must be different for each run.')
flags.DEFINE_string('init_model', None,
                    'Checkpoint path of the model to initialize with.')
flags.DEFINE_integer('save_every', 1000,
                     'Average period of steps after which we save a model.')
# Optimization
flags.DEFINE_float('proj_weight', 10, 'Weighting factor for projection loss.')
flags.DEFINE_float('volume_weight', 0, 'Weighting factor for volume loss.')
flags.DEFINE_float('viewpoint_weight', 1, 'Weighting factor for viewpoint loss.')
flags.DEFINE_float('learning_rate', 0.0001, 'Learning rate.')
flags.DEFINE_float('weight_decay', 0.001, 'Weight decay parameter while training.')
flags.DEFINE_float('clip_gradient_norm', 0, 'Gradient clim norm, leave 0 if no gradient clipping.')
flags.DEFINE_integer('max_number_of_steps', 10000, 'Maximum number of steps for training.')
# Summary
flags.DEFINE_integer('save_summaries_secs', 15, 'Seconds interval for dumping TF summaries.')
flags.DEFINE_integer('save_interval_secs', 60 * 5, 'Seconds interval to save models.')

# Scheduling
flags.DEFINE_string('master', '', 'The address of the tensorflow master')
flags.DEFINE_bool('sync_replicas', False, 'Whether to sync gradients between replicas for optimizer.')
flags.DEFINE_integer('worker_replicas', 1, 'Number of worker replicas (train tasks).')
flags.DEFINE_integer('backup_workers', 0, 'Number of backup workers.')
flags.DEFINE_integer('ps_tasks', 0, 'Number of ps tasks.')
flags.DEFINE_integer('task', 0,
                     'Task identifier flag to be set for each task running in distributed manner. Task number 0 '
                     'will be chosen as the chief.')

FLAGS = flags.FLAGS


def main(_):
  train_dir = os.path.join(FLAGS.checkpoint_dir, FLAGS.model_name, 'train')
  save_image_dir = os.path.join(train_dir, 'images')
  if not os.path.exists(train_dir):
    os.makedirs(train_dir)
  if not os.path.exists(save_image_dir):
    os.makedirs(save_image_dir)

  g = tf.Graph()
  with g.as_default():
    with tf.device(tf.train.replica_device_setter(FLAGS.ps_tasks)):
      global_step = slim.get_or_create_global_step()
      ###########
      ## model ##
      ###########
      model = model_ptn.model_PTN(FLAGS)
      ##########
      ## data ##
      ##########
      train_data = model.get_inputs(
          FLAGS.inp_dir,
          FLAGS.dataset_name,
          'train',
          FLAGS.batch_size,
          FLAGS.image_size,
          FLAGS.vox_size,
          is_training=True)
      inputs = model.preprocess(train_data, FLAGS.step_size)
      ##############
      ## model_fn ##
      ##############
      model_fn = model.get_model_fn(
          is_training=True, reuse=False, run_projection=True)
      outputs = model_fn(inputs)
      ##################
      ## train_scopes ##
      ##################
      if FLAGS.init_model:
        train_scopes = ['decoder']
        init_scopes = ['encoder']
      else:
        train_scopes = ['encoder', 'decoder']

      ##########
      ## loss ##
      ##########
      task_loss = model.get_loss(inputs, outputs)

      regularization_loss = model.get_regularization_loss(train_scopes)
      loss = task_loss + regularization_loss
      ###############
      ## optimizer ##
      ###############
      optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
      if FLAGS.sync_replicas:
        optimizer = tf.train.SyncReplicasOptimizer(
            optimizer,
            replicas_to_aggregate=FLAGS.workers_replicas - FLAGS.backup_workers,
            total_num_replicas=FLAGS.worker_replicas)

      ##############
      ## train_op ##
      ##############
      train_op = model.get_train_op_for_scope(loss, optimizer, train_scopes)
      ###########
      ## saver ##
      ###########
      saver = tf.train.Saver(max_to_keep=np.minimum(5,
                                                    FLAGS.worker_replicas + 1))

      #############
      ## init_fn ##
      #############
      if FLAGS.init_model:
        init_fn = model.get_init_fn(init_scopes)
      else:
        init_fn = None

      ##############
      ## training ##
      ##############
      import time
      time_start=time.time()
      slim.learning.train(
          train_op=train_op,
          logdir=train_dir,
          init_fn=init_fn,
          master=FLAGS.master,
          is_chief=(FLAGS.task == 0),
          number_of_steps=FLAGS.max_number_of_steps,
          saver=saver,
          save_summaries_secs=FLAGS.save_summaries_secs,
          save_interval_secs=FLAGS.save_interval_secs)
      print("train_time=",time.time()-time_start)


if __name__ == '__main__':
  app.run()
