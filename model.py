#! /usr/bin/env python

import tensorflow as tf
import yaml
from option import opts
import itertools
import numpy as np
import math

class AttnNet(object):
    """
        This code is inspired by google's paper Attention Is All You Need
        Build a transformer to predict the system energy
        Developped after DensityNet (which seems not working) 
    """

    def __init__(self, max_atom, n_atom_embed, n_kmesh, n_trans, n_heads, n_ff):
        self.atom_kinds = 120 # index zero is unkown atoms
        self.max_atom = max_atom
        self.n_atom_embed = n_atom_embed
        # Reciprocal points
        self.n_kmesh = n_kmesh
        self.n_trans = n_trans
        self.n_heads = n_heads

        self.n_pos_embed = 2*n_kmesh**3
        self.n_dims = 2*self.n_kmesh**3+self.n_atom_embed

        self.n_ff = n_ff
        self._build_graph()

    def get_atom_embed(self, atom_ids):
        with tf.variable_scope("atomic_embedding"):
            embed=tf.get_variable("embed", shape=(self.atom_kinds, self.n_atom_embed),
                dtype=tf.float32, initializer=tf.initializers.random_normal(stddev=1.0e-3))
            embeds = tf.nn.embedding_lookup(embed, atom_ids)
        return embeds

    def get_k_mesh(self):
        with tf.variable_scope("reciprocal_mesh"):
            kmesh = []
            for i, j, k in itertools.product(*itertools.repeat(range(self.n_kmesh), 3)):
                kmesh.append(np.array([i/self.n_kmesh, j/self.n_kmesh, k/self.n_kmesh]))

            kmesh = np.stack(kmesh, axis=0) # n_kmesh**3 x 3
        return tf.constant(kmesh, dtype=tf.float32)              
        
    def get_pos_embed(self, latts, carts):
        with tf.variable_scope("position_embedding"):
            inv_latts = tf.matrix_inverse(latts)
            kmesh = self.get_k_mesh()
            rvec = tf.einsum("nl,kml->knm", kmesh, inv_latts) # nbatch x n_kmesh**3 x 3
            phase = 2*math.pi*tf.einsum("knl,kml->knm", carts, rvec) # nbatch x max_atom x n_kmesh**3
            s_phase = tf.sin(phase)  
            c_phase = tf.cos(phase)
        return tf.concat((s_phase, c_phase), axis=-1) # nbatch x max_atom x 2*n_kmesh**3

    def get_trans_param(self, name=""):
        with tf.variable_scope("parameters_"+name):
            trans=tf.get_variable("trans", shape=(self.n_dims, self.n_trans),
                dtype=tf.float32, initializer=tf.initializers.random_normal(stddev=1.0e-3))
        return trans 


    def attn(self, embed, mask, name=""):
        """
            Assume embeddiing has nbatch x max_atom x n_dims
            mask has nbatch x max_atom
        """
        with tf.variable_scope("attn_"+name):
            K = self.get_trans_param("K")
            Q = self.get_trans_param("Q")
            V = self.get_trans_param("V")
            kdata = tf.einsum("nml,lk->nmk", embed, K)
            qdata = tf.einsum("nml,lk->nmk", embed, Q)
            vdata = tf.einsum("nml,lk->nmk", embed, V) # nbatch x max_atom x n_trans
            kq = tf.einsum("nml,nkl->nmk", qdata, kdata)*(1/math.sqrt(self.n_trans)) # nbatch x max_atom x max_atom
            #mask = tf.expand_dims(mask, 1) # nbatch x 1 x max_atom
            mask = tf.keras.backend.repeat(mask, self.max_atom) 
            score = tf.where(mask, -9999*tf.ones_like(kq), kq)
            #score = kq
            #score = tf.scatter_update(tf.Variable(kq, validate_shape=False), mask, -9999)# assign a large number
            w = tf.nn.softmax(score, axis=-1) # calculate attention weight, nbatch x max_atom x max_atom
            vout = tf.einsum("nml,nlk->nmk", w, vdata) # nbatch x max_atom x n_trans
        return vout

    def trans(self, features, masks2, name=""):
        with tf.variable_scope("transformer_{}".format(name)):
            attns = []
            for idx in range(self.n_heads):
                attns.append(self.attn(features, masks2, "{}".format(idx)))
            attns = tf.concat(attns, axis=-1) # nbatch x max_atom x (n_heads x n_trans)

            # Feed-forward neural networks to calculate energy
            layer1 = tf.contrib.layers.layer_norm(features+attns)
            layer2 = tf.keras.layers.Dense(self.n_ff, activation=tf.nn.relu)(layer1)
            layer3 = tf.keras.layers.Dense(self.n_dims, activation=None)(layer2)
        return layer3

    def stru_factor(self, atom_ids, latts, coords):
        """
            Get structure factor of whole crystal
        """
        with tf.variable_scope("structure_factor"):
            atom_embeds = self.get_atom_embed(atom_ids)

            # mask out padding atoms
            coeff = tf.keras.layers.Dense(1, activation=tf.nn.tanh)(atom_embeds) # nbatch x maxatom x 1
            coeff = tf.where(tf.expand_dims(tf.not_equal(atom_ids, 0), axis=-1), coeff, tf.zeros_like(coeff))
            coeff_sq = tf.expand_dims(tf.complex(tf.einsum("bij,bkj->bik", coeff, coeff), 0.0), -1)

            inv_latts = tf.matrix_inverse(latts)
            kmesh = self.get_k_mesh()
            rvec = 2*math.pi*tf.einsum("nl,kml->knm", kmesh, inv_latts) # nbatch x n_kmesh**3 x 3
            phase = tf.einsum("bik,bjk->bij", coords, rvec) # nbatch x natom x n_kmesh**3
            wfunc = tf.exp(tf.complex(0.0, -phase)) # nbatch x natom x n_kmesh**3
            wfunc1 = tf.expand_dims(wfunc, 1)
            wfunc2 = tf.expand_dims(wfunc, 2)
            sq = tf.reduce_sum(wfunc1*tf.math.conj(wfunc2)*coeff_sq, axis=2)
            cf = tf.math.real(sq)/tf.reduce_sum(tf.square(coeff), axis=[1], keepdims=True) 
            sf = tf.math.imag(sq)/tf.reduce_sum(tf.square(coeff), axis=[1], keepdims=True) 

            feat = tf.concat([atom_embeds, cf, sf], axis = -1)
            #coeff_ij = tf.einsum("bij,bkj->bik", coeff, coeff) # nbatch x maxatom x maxatom
            #coeff_ij = tf.expand_dims(coeff_ij, axis=-1) # nbatch x maxatom x maxatom x 1
            #coeff_ij = tf.expand_dims(coeff_ij, axis=1) # nbatch x 1 x maxatom x maxatom x 1
            #rij = tf.expand_dims(coords, dim=1) - tf.expand_dims(coords, dim=2) # nbatch x maxatom x maxatom x 3
            #rij = tf.expand_dims(rij, axis=1) # nbatch x 1 x maxatom x maxatom x 3
            #
            #rvec = tf.reshape(rvec, shape=(-1, n_kmesh**3, 1, 1, 3)) # batch x n_kmesh**3 x 1 x 1 x 3
            #
            #sf =  tf.reduce_sum(rij*rvec, axis=[2,3,4])/tf.reuce_sum(tf.square(coeff), axis=[1])

        return feat


    def energy_func(self, atom_ids, coords, latts):

        with tf.variable_scope("attention"):

            #atom_embeds = self.get_atom_embed(atom_ids)
            #pos_embeds = self.get_pos_embed(latts, coords)
            #embeds = tf.concat((atom_embeds, pos_embeds), axis=-1)
            embeds = self.stru_factor(atom_ids, latts, coords)

            masks1 = tf.cast(tf.not_equal(atom_ids, 0), tf.float32)
            masks2 = tf.equal(atom_ids, 0)

            layer1 = self.trans(embeds, masks2, "1")
            #layer2 = self.trans(layer1, masks2, "2")
            #layer3 = self.trans(layer2, masks2, "3")

            layer4 = tf.keras.layers.Dense(self.n_ff, activation=tf.nn.relu)(layer1)
            layer5 = tf.keras.layers.Dense(self.n_ff, activation=tf.nn.relu)(layer4)
            layer6 = tf.squeeze(tf.keras.layers.Dense(1, activation=None)(layer5))
            atomic_energy = masks1*layer6
            energy = tf.reduce_sum(atomic_energy, axis=-1)

            ## mask out unused atoms
            #layer4 = tf.expand_dims(masks1, axis=-1)*layer3 # nbatch x maxatom x self.n_dims
            ## nbatch x self.n_dims, average over all usable atoms
            #layer5 = tf.reduce_sum(layer4, axis=1)/tf.reduce_sum(masks1, axis=-1, keepdims=True)
            #layer6 = tf.keras.layers.Dense(self.n_ff, activation=tf.nn.relu)(layer5)
            #layer7 = tf.keras.layers.Dense(1, activation=None)(layer6)
            #energy = tf.squeeze(layer7)

        return energy
            
    def _build_graph(self):
        self.train_graph = tf.Graph()
        self.valid_graph = tf.Graph()
        self.infer_graph = tf.Graph()

    def train(self, atom_ids, coords, latts, force, energy, stress):
        with self.train_graph.as_default():
            energy_p = self.energy_func(atom_ids, coords, latts)
            
            masks1 = tf.cast(tf.not_equal(atom_ids, 0), tf.float32)
            masks1 = tf.expand_dims(masks1, axis=-1)

            force_p = -tf.gradients(energy_p, coords)[0]*masks1
            vols = tf.reshape(tf.abs(tf.linalg.det(latts)), (-1, 1, 1))
            stress_p = -tf.einsum("bij,bkj->bik", tf.gradients(energy_p, latts)[0], latts)/vols

            masks = tf.cast(tf.not_equal(atom_ids, 0), tf.float32)
            na = tf.reduce_sum(masks, axis=1)
            self.avg_na =  tf.reduce_mean(na)
            #loss = tf.losses.huber_loss(energy/na, energy_p/na, delta=1.0) + \
            #       tf.losses.huber_loss(force, force_p, delta=1.0) + \
            #       tf.losses.huber_loss(stress, stress_p, delta=0.1)

            loss = tf.losses.mean_squared_error(energy/na, energy_p/na) + \
                   tf.losses.mean_squared_error(force, force_p) + \
                   tf.losses.mean_squared_error(stress, stress_p)

            energy_loss_t = tf.sqrt(tf.reduce_mean(((energy-energy_p)/na)**2))
            #na = tf.reshape(na, (-1, 1, 1))
            force_loss_t = tf.sqrt(tf.reduce_mean(tf.reduce_sum(((force-force_p))**2, axis=-1)))
            stress_loss_t = tf.sqrt(tf.reduce_mean(((stress-stress_p))**2))

        return loss, energy_loss_t, force_loss_t, stress_loss_t

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
