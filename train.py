#! /usr/bin/env python

import tensorflow as tf
import yaml
from model import DensityNet

from tensorflow.python import debug as tf_debug


def _parse_function(example, max_atom):
    features = {
        "na"    : tf.FixedLenFeature([], tf.int64),
        "energy": tf.FixedLenFeature([], tf.float32),
        "serial": tf.FixedLenFeature([max_atom], tf.int64),
        "latt"  : tf.FixedLenFeature([3, 3], tf.float32),
        "coord":  tf.FixedLenFeature([max_atom, 3], tf.float32),
        "stress": tf.FixedLenFeature([3, 3], tf.float32) ,
        "force" : tf.FixedLenFeature([max_atom, 3], tf.float32), }
    parsed_features = tf.parse_single_example(example, features) 
    #parsed_features["coords"] = tf.reshape(parsed_features["coords"], (max_atom, 3))
    #parsed_features["force"] = tf.reshape(parsed_features["force"], (max_atom, 3))
    #parsed_features["latt"] = tf.reshape(parsed_features["latt"], (3, 3))
    #parsed_features["stress"] = tf.reshape(parsed_features["stress"], (3, 3))
    return parsed_features

class ModelTrainer(object):
    """
        This class is aimed at training DensityNet model
    """

    def __init__(self, config):
        self.config = yaml.load(open(config))
        self.max_atom = self.config["max_atom_num"]
        self.embed_size = self.config["embed_size"]
        self.coeff_size = self.config["coeff_size"]
        self.nsc = self.config["super_cell"]
        self.train_path = self.config["train_file"]
        self.nepoch = self.config["nepoch"]
        self.bs = self.config["batch_size"]
        self.lr = self.config["learning_rate"]
        self.mean = self.config["mean"]
        self.std = self.config["std"]
        self.density_net = DensityNet(self.max_atom, self.embed_size, self.coeff_size, self.nsc, self.mean, self.std)

    def train(self):

        with self.density_net.train_graph.as_default():

            dataset = tf.data.TFRecordDataset(tf.constant([self.train_path]))
            dataset = dataset.map(lambda x: _parse_function(x, self.max_atom))
            dataset = dataset.shuffle(buffer_size=1000)
            dataset = dataset.batch(self.bs)
            dataset = dataset.repeat(self.nepoch)
            iterator = dataset.make_one_shot_iterator()
            features = iterator.get_next() 
      
            loss_t, energy_loss_t, force_loss_t, stress_loss_t = \
                self.density_net.train(features["serial"], features["coord"], features["latt"], 
                    features["force"], features["energy"], features["stress"])
       
            global_step = tf.Variable(0, name='global_step',trainable=False)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                optim = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(loss_t, global_step=global_step)
            writer = tf.summary.FileWriter("./logs", self.density_net.train_graph)
            tf.summary.scalar("0.loss", loss_t)
            tf.summary.scalar("1.energy_loss", energy_loss_t)
            tf.summary.scalar("2.force_loss", force_loss_t)
            tf.summary.scalar("3.stress_loss", stress_loss_t)
            ms_op = tf.summary.merge_all()
            check_op = tf.add_check_numerics_ops()
            with tf.train.MonitoredTrainingSession() as s:
                #s = tf_debug.LocalCLIDebugWrapperSession(s)
                #s.run(tf.global_variables_initializer())
                #s.run(iterator.initializer)
                while not s.should_stop():
                    #_, loss, _, ms, step = s.run([check_op, loss_t, optim, ms_op, global_step])
                    #_, loss, _, ms, step = s.run([self.density_net.test_ops, loss_t, optim, ms_op, global_step])
                    loss, _, ms, step = s.run([loss_t, optim, ms_op, global_step])
                    writer.add_summary(ms, step)
                    print("Current iteration: {:5d}, Current loss: {:.2f}".format(step, loss))
                print("Fininshed training...")


if __name__ == "__main__":
    mt = ModelTrainer("./config.yml")
    mt.train()
