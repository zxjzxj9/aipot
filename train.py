#! /usr/bin/env python

import tensorflow as tf
import yaml
from model import AttnNet

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

        self.n_atom_embed = self.config["n_atom_embed"]
        self.n_kmesh = self.config["n_kmesh"]
        self.n_trans = self.config["n_trans"]
        self.n_heads = self.config["n_heads"]
        self.n_ff = self.config["n_ff"]

        self.train_path = self.config["train_file"]
        self.nepoch = self.config["nepoch"]
        self.bs = self.config["batch_size"]
        self.lr = self.config["learning_rate"]
        self.model_path = self.config.get("model_path")

        self.attn_net = AttnNet(self.max_atom, self.n_atom_embed, self.n_kmesh, self.n_trans, self.n_heads, self.n_ff)

    def train(self):

        with self.attn_net.train_graph.as_default():

            dataset = tf.data.TFRecordDataset(tf.constant([self.train_path]))
            dataset = dataset.map(lambda x: _parse_function(x, self.max_atom))
            dataset = dataset.shuffle(buffer_size=1000)
            dataset = dataset.batch(self.bs)
            dataset = dataset.repeat(self.nepoch)
            iterator = dataset.make_one_shot_iterator()
            features = iterator.get_next() 
      
            loss_t, energy_loss_t, force_loss_t, stress_loss_t = \
                self.attn_net.train(features["serial"], features["coord"], features["latt"], 
                    features["force"], features["energy"], features["stress"])
       
            global_step = tf.Variable(0, name='global_step',trainable=False)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                #optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.99, beta2=0.999)
                optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
                #optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, clip_norm=5.0)
                optim = optimizer.minimize(loss_t, global_step=global_step)
            #print(tf.trainable_variables()); import sys; sys.exit()
            writer = tf.summary.FileWriter("./logs", self.attn_net.train_graph)
            tf.summary.scalar("0.loss", loss_t)
            tf.summary.scalar("1.energy_loss", energy_loss_t)
            tf.summary.scalar("2.force_loss", force_loss_t)
            tf.summary.scalar("3.stress_loss", stress_loss_t)
            ms_op = tf.summary.merge_all()
            check_op = tf.add_check_numerics_ops()

            saver = tf.train.Saver()
            #with tf.train.MonitoredTrainingSession() as s:
            with tf.Session() as s:
                #s = tf_debug.LocalCLIDebugWrapperSession(s)
                s.run(tf.global_variables_initializer())

                if self.model_path:
                    print("Loading model at:", self.model_path)
                    saver.restore(s, self.model_path)
                #s.run(iterator.initializer)
                #while not s.should_stop():
                while True:
                    try:    
                        #_, loss, _, ms, step = s.run([check_op, loss_t, optim, ms_op, global_step])
                        #_, loss, _, ms, step = s.run([self.density_net.test_ops, loss_t, optim, ms_op, global_step])
                        #avg_na = s.run(tf.reduce_sum(tf.cast(tf.not_equal(features["serial"], 0), tf.float32)))/self.bs
                        loss, _, ms, step, avg_na = s.run([loss_t, optim, ms_op, global_step, self.attn_net.avg_na])
                        writer.add_summary(ms, step)
                        #print("Current iteration: {:5d}, Current loss: {:.2f}".format(step, loss))
                        print("Current iteration: {:5d}, Current loss: {:.2f}, avg_na: {:.2f}".format(step, loss, avg_na))
                        if step % 1000 == 0:
                            print("Save model at: {}".format(saver.save(s, "./models/model.ckpt")))
                    except tf.errors.OutOfRangeError as e:
                        print("Fininshed training...")
                        break


if __name__ == "__main__":
    mt = ModelTrainer("./config.yml")
    mt.train()
