from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging
from engine import train

import tensorflow.compat.v1 as tf
from tensorflow_gan.examples.mnist import train_lib


flags.DEFINE_integer('batch_size', 32, 'The number of images in each batch.')

flags.DEFINE_string('train_log_dir', '/tfgan_logdir/mnist',
                    'Directory where to write event logs.')

flags.DEFINE_integer('max_number_of_steps', 20000,
                     'The maximum number of gradient steps.')

flags.DEFINE_string('gan_type', 'unconditional',
                    'Either `unconditional`, `conditional`, or `infogan`.')

flags.DEFINE_integer('grid_size', 5, 'Grid size for image visualization.')

flags.DEFINE_integer('noise_dims', 64,
                     'Dimensions of the generator noise vector.')

FLAGS = flags.FLAGS

def main(_):
  hparams = train_lib.HParams(FLAGS.batch_size, FLAGS.train_log_dir,
                              FLAGS.max_number_of_steps, FLAGS.gan_type,
                              FLAGS.grid_size, FLAGS.noise_dims)
  train_lib.train(hparams)

if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  tf.disable_v2_behavior()
  app.run(main)


