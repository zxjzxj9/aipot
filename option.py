#! /usr/bin/env python

import tensorflow as tf

opts = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("nepochs", 100, "Training epoch number")
tf.app.flags.DEFINE_integer("nbatch", 128, "Batch data number")
tf.app.flags.DEFINE_float("scale_f", 1.0, "Scale factor of force")
tf.app.flags.DEFINE_float("scale_s", 1.0, "Scale factor of stress")

tf.app.flags.DEFINE_integer("nembeds", 128, "Atomic embedding size")
