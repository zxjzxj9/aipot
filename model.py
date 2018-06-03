#! /usr/bin/env python

import tensorflow as tf
import yaml
from option import opts

def _parse_function(example):
    features = {
        "na"    : tf.FixedLenFeature([], tf.int64),
        "energy": tf.FixedLenFeature([], tf.float32),
        "serial": tf.FixedLenSequenceFeature([], tf.int64, True, 0),
        "latt"  : tf.FixedLenSequenceFeature([], tf.float32, True, 0.0),
        "coords": tf.FixedLenSequenceFeature([], tf.float32, True, 0.0),
        "stress": tf.FixedLenSequenceFeature([], tf.float32, True, 0.0),
        "force" : tf.FixedLenSequenceFeature([], tf.float32, True, 0.0), }
    parsed_features = tf.parse_single_example(example, features) 
    parsed_features["coords"] = tf.reshape(parsed_features["coords"], (-1, 3))
    parsed_features["force"] = tf.reshape(parsed_features["force"], (-1, 3))
    parsed_features["latt"] = tf.reshape(parsed_features["latt"], (3, 3))
    parsed_features["stress"] = tf.reshape(parsed_features["stress"], (3, 3))
    return parsed_features

class PESModel(object):
    def __init__(self, config="./config.yml"):
        self.train_graph = tf.Graph()
        # validation and test graph are same
        self.val_graph = tf.Graph()
        self.config = yaml.load(open(config))
        # self.test_graph = tf.Graph()
        self.infer_graph = tf.Graph()
        self._build_model()

    def _build_model(self):
        with self.train_graph.as_default():
            self.train_loss = self.eval_loss(self.cofig["train_file"])
        
        with self.val_graph.as_default():
            self.val_loss = self.eval_loss(self.config["val_file"])

        with self.infer_graph.as_default():
            na = tf.placeholder(dtype=tf.int64, shape=(None,))
            ntyp = tf.placeholder(dtype=tf.int64, shape=(None, self.config["max_atom_num"]))
            latt = tf.placeholder(dtype=tf.float32, shape=(None, 3, 3))
            coords = tf.placeholder(dtype=tf.float32, shape=(None, self.config["max_atom_num"], 3))
            self.pes = calc_pes(na, ntyp, latt, coords)
            
    def eval_loss(self, data_path):
        dataset = tf.data.TFRecordDataset(tf.constant([data_path]))
        dataset = dataset.map(_parse_function)
        dataset.repeat(opts.nepochs)
        dataset = dataset.batch(opts.nbatch)
        iterator = dataset.make_initializable_iterator()
        train_data = iterator.get_next()

        mask = tf.sequence_mask(train_data["na"], 
            maxlen=self.config["max_atom_num"], dtype=tf.float32)

        e_pred, f_pred, s_pred = self.calc_pes(
            train_data["na"], train_data["serial"], train_data["latt"], 
            train_data["coords"])

        loss = tf.reduce_mean(tf.square(train_data["energy"] - e_pred)) + \
                   opts.scale_f*tf.reduce_mean(mask*tf.square(train_data["force"] - f_pred)) + \
                   opts.scale_s*tf.reduce_mean(mask*tf.square(train_data["stress"] - s_pred))

    def calc_pes(self, na, ntyp, latt, coords):
        """
        input:
            na :     nbatch,                    int,   atom number
            ntyp:    nbatch x max_atom_num,     int,   species
            latt:    nbatch x 3 x 3,            float, lattice vector
            coords:   nbatch x max_atom_num x 3, float, atomic coordinates
        output:
            energy:  nbatch,                    float, energy 
            force:   nbatch x max_atom_num x 3, float, atomic forces
            stress:  nbatch x 6 stresses,       float, lattice stress
        """
       
        with tf.variable_scope("embedding", initializer=tf.contrib.layers.xavier_initializer()):
            embeds = tf.get_variable("embedding", shape=[len(self.config["embeds"]), opts.nembeds], dtype=tf.float32)
        
           
        


    def train(self):
        """
            Traing the model
        """
        pass


if __name__ == "__main__":
    data_path = "./data/train.tfrecords"
    #filename_queue = tf.train.string_input_producer([data_path], num_epochs=1)
    #features = tf.parse_single_example(serialized_example, features=feature)
    dataset = tf.data.TFRecordDataset(tf.constant([data_path]))
    dataset = dataset.map(_parse_function)
    dataset.repeat(1)
    dataset = dataset.batch(32)
    iterator = dataset.make_initializable_iterator()

    with tf.Session() as s:
        s.run(tf.global_variables_initializer())
        s.run(iterator.initializer)
        while True:
            print(s.run(iterator.get_next()))
            break

    