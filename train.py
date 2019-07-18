#! /usr/bin/env python

import tensorflow as tf
import yaml
from model import AttnNet

import numpy as np

#import memory_saving_gradients
# monkey patch tf.gradients to point to our custom version, with automatic checkpoint selection
#tf.__dict__["gradients"] = memory_saving_gradients.gradients_memory

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
        self.n_zetas = self.config["n_zetas"]

        self.train_path = self.config["train_file"]
        self.valid_path = self.config["valid_file"]
        self.nepoch = self.config["nepoch"]
        self.bs = self.config["batch_size"]
        self.lr = self.config["learning_rate"]
        self.model_path = self.config.get("model_path")

        self.attn_net = AttnNet(self.max_atom, self.n_atom_embed, self.n_kmesh, self.n_trans, self.n_heads, self.n_ff, self.n_zetas)
        self.valid_ops = self._get_valid_ops()
        self.train_ops = self._get_train_ops()

        #config = tf.ConfigProto()
        #config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

        self.train_session = tf.Session(graph=self.attn_net.train_graph) #, config=config)

        saver = self.train_saver
        s = self.train_session
        s.run(self.train_init)

        if self.model_path:
            print("Before training, loading model at:", self.model_path)
            saver.restore(s, self.model_path)
        self.valid_session = tf.Session(graph=self.attn_net.valid_graph) #, config=config)


    def _get_valid_ops(self):

        with self.attn_net.valid_graph.as_default():

            dataset = tf.data.TFRecordDataset(tf.constant([self.valid_path]))
            dataset = dataset.map(lambda x: _parse_function(x, self.max_atom))
            dataset = dataset.batch(self.bs)
            dataset = dataset.repeat(1)
            #iterator = dataset.make_one_shot_iterator()
            iterator = dataset.make_initializable_iterator()
            features = iterator.get_next()

            loss_t, energy_loss_t, force_loss_t, stress_loss_t = \
                self.attn_net.validate(features["serial"], features["coord"], features["latt"],
                    features["force"], features["energy"], features["stress"])
            global_step = tf.Variable(0, name='global_step',trainable=False)

            tf.summary.scalar("0.loss", loss_t)
            tf.summary.scalar("1.energy_loss", energy_loss_t)
            tf.summary.scalar("2.force_loss", force_loss_t)
            tf.summary.scalar("3.stress_loss", stress_loss_t)

            ms_op = tf.summary.merge_all()
            self.valid_init = tf.global_variables_initializer()
            self.valid_saver = tf.train.Saver()
            self.valid_iterator = iterator
            self.valid_writer = tf.summary.FileWriter("./valid_logs", self.attn_net.valid_graph)
            return ms_op, global_step, [loss_t, energy_loss_t, force_loss_t, stress_loss_t]

    def valid(self, model_path = "./models/model.ckpt"):

        writer = self.valid_writer
        saver = self.valid_saver
        s = self.valid_session
        s.run(self.valid_init)
        s.run(self.valid_iterator.initializer)
        pdata = [0.0, 0.0, 0.0, 0.0]
        cnt = 0


        #if self.model_path:
        #    print("Loading model at:", self.model_path)
        #    saver.restore(s, self.model_path)
        # model_path = "./models/model.ckpt"
        saver.restore(s, model_path)

        while True:
            try:
                ms, step, pdatat = s.run(self.valid_ops)
                writer.add_summary(ms, step)
                pdata[0] += pdatat[0].sum()
                pdata[1] += pdatat[1].sum()
                pdata[2] += pdatat[2].sum()
                pdata[3] += pdatat[3].sum()
                cnt += 1
            except tf.errors.OutOfRangeError as e:
                print("Avg Loss: {:.3f}, Avg Energy RMS: {:.3f}, Avg Force RMS: {:.3f}, Avg Stress RMS: {:.3f}".format(
                    pdata[0]/cnt, pdata[1]/cnt, pdata[2]/cnt, pdata[3]/cnt))
                #print("Fininshed training...")
                break

    def _get_train_ops(self):

        with self.attn_net.train_graph.as_default():

            dataset = tf.data.TFRecordDataset(tf.constant([self.train_path]))
            dataset = dataset.map(lambda x: _parse_function(x, self.max_atom))
            dataset = dataset.shuffle(buffer_size=200000)
            dataset = dataset.batch(self.bs)
            #dataset = dataset.repeat(self.nepoch)
            dataset = dataset.repeat(1)
            #iterator = dataset.make_one_shot_iterator()
            iterator = dataset.make_initializable_iterator()
            features = iterator.get_next()

            loss_t, energy_loss_t, force_loss_t, stress_loss_t = \
                self.attn_net.train(features["serial"], features["coord"], features["latt"],
                    features["force"], features["energy"], features["stress"])

            global_step = tf.Variable(0, name='global_step',trainable=False)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                #optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.99, beta2=0.999)
                optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
                optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, clip_norm=1.0)
                optim = optimizer.minimize(loss_t, global_step=global_step)
            #print(tf.trainable_variables()); import sys; sys.exit()
            tf.summary.scalar("0.loss", loss_t)
            tf.summary.scalar("1.energy_loss", energy_loss_t)
            tf.summary.scalar("2.force_loss", force_loss_t)
            tf.summary.scalar("3.stress_loss", stress_loss_t)

            for var in tf.trainable_variables():
                tf.summary.histogram(var.op.name, var)


            #print(dir(optim.inputs))
            #print(optim.inputs())
            #import sys; sys.exit()
            #grads_vars = tf.gradients(loss_t, tf.trainable_variables())
            #for var in grads_vars:
            #    tf.summary.histogram(var.op.name, var)
            #for grad, var in grads_vars:
            #    if grad is not None:
            #        tf.summary.histogram(var.op.name + '/gradients', grad)

            ms_op = tf.summary.merge_all()
            #check_op = tf.add_check_numerics_ops()
            self.train_init = tf.global_variables_initializer()
            self.train_saver = tf.train.Saver()
            self.train_iterator = iterator
            self.train_writer = tf.summary.FileWriter("./logs", self.attn_net.train_graph)
            return loss_t, optim, ms_op, global_step, self.attn_net.avg_na, self.attn_net.check_ops
            #return check_op, loss_t, optim, ms_op, global_step, self.attn_net.avg_na, self.attn_net.check_ops

    def train(self):

        writer = self.train_writer
        saver = self.train_saver

        s = self.train_session
        #s.run(self.train_init)
        s.run(self.train_iterator.initializer)

        while True:
            try:
                loss, _, ms, step, avg_na, ret = s.run(self.train_ops)
                #_, loss, _, ms, step, avg_na, ret = s.run(self.train_ops)
                writer.add_summary(ms, step)
                print("Current iteration: {:5d}, Current loss: {:.2f}, avg_na: {:.2f}".format(step, loss, avg_na), end="\r")
                #import sys; sys.exit()
            except tf.errors.OutOfRangeError as e:
                print("")
                #print("\nFininshed training...")
                break
        #print("")
        print("Save model at: {}".format(saver.save(s, "./models/model.ckpt")))

    def train_and_eval(self):
        tf.logging.set_verbosity(tf.logging.ERROR)
        print("###")

        for i in range(self.nepoch):
            print("In epoch #{}, ".format(i+1), end="\n")
            self.train()
            #self.valid()
            if (i+1)%10==0: self.valid()

        self.train_session.close()
        self.valid_session.close()

if __name__ == "__main__":
    mt = ModelTrainer("./config.yml")
    mt.train_and_eval()
    #mt.train()
    #mt.valid()
    #mt.valid()
    #for i in range(100):
    #    print("In epoch #{}, ".format(i+1), end="")
    #    mt.train()
