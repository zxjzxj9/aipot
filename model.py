#! /usr/bin/env python

import tensorflow as tf

class PESModel(object):
    def __init__(self):
        self.train_graph = tf.Graph()
        self.val_graph = tf.Graph()
        self.test_graph = tf.Graph()
        self._build_model()

    def _build_model(self):
        with self.train_graph.as_default():
            pass

    def train(self, na, atom_type, lattvec, coord, energy, force, stress):
        """
            Traing the model
            input: 
                na : nbatch, type int, atom number
                atom_type: nbatch x max_atom_num, type int, species
                lattvec: nbatch x 3 x 3, type float, lattice vector
                coord: nbatch x max_atom_num x 3, atomic coordinates
                energy: nbatch, type float, energy 
                force: nbatch x max_atom_num x 3, atomic forces
                stress: nbatch x 6 stresses
        """
        pass
