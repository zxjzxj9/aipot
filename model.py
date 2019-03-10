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
        self.n_zetas = 256
        self.nsc = 2

        #self.n_pos_embed = 2*n_kmesh**3
        #self.n_dims = 2*self.n_kmesh**3+self.n_atom_embed
        self.n_dims = self.n_zetas + self.n_atom_embed

        self.n_ff = n_ff
        self._build_graph()

    def get_atom_embed(self, atom_ids):
        with tf.variable_scope("atomic_embedding"):
            embed=tf.get_variable("embed", shape=(self.atom_kinds, self.n_atom_embed),
                dtype=tf.float32, initializer=tf.initializers.random_normal(stddev=1.0e-3))
            embeds = tf.nn.embedding_lookup(embed, atom_ids)
        return embeds

    def get_param_embed(self, atom_ids, name, mean=5.0):
        with tf.variable_scope("{}_embedding".format(name)):
            embed=tf.get_variable("embed", shape=(self.atom_kinds, self.n_zetas),
                dtype=tf.float32, initializer=tf.initializers.random_normal(mean=mean, stddev=1.0e-3*abs(mean)))
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
            # print(attns); import sys; sys.exit()
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

        return feat

    def get_sc(self):
        """
            get supercell vector given the grid mesh, dim (nsc**3)*3
        """
        sc = []
        a = np.array([1, 0, 0], dtype=np.float32)
        b = np.array([0, 1, 0], dtype=np.float32)
        c = np.array([0, 0, 1], dtype=np.float32)
        # order z -> y -> x
        for i, j, k in itertools.product(*itertools.repeat(range(-(self.nsc//2),self.nsc//2+1), 3)):
            sc.append(i*a + j*b + k*c)
        sc = np.stack(sc, axis=0)
        #print(sc.shape)

        with tf.variable_scope("supercell"):
            sc = tf.constant(sc, dtype=tf.float32)
        return sc

    def dist_weight(self, atom_ids, latts, coords):

        with tf.variable_scope("distance_weight"):
            atom_embeds = self.get_atom_embed(atom_ids)

            zeta = self.get_param_embed(atom_ids, "zeta", 0.1)   # batch x maxatom x nzeta    
            zeta = tf.expand_dims(zeta, axis=1) # batch x 1 x maxatom x nzeta

            coeff = self.get_param_embed(atom_ids, "coeff", 0.001) # batch x maxatom x nzeta
            mask = tf.cast(tf.not_equal(atom_ids, 0), tf.float32) # batch x maxatom
            mask = tf.expand_dims(mask, -1)
            coeff = mask*coeff
            coeff = tf.expand_dims(coeff, axis=1) # batch x 1 x maxatom x nzeta

            inv_latts = tf.matrix_inverse(latts)
            frac = tf.einsum("bij,bjk->bik", coords, inv_latts) # batch x maxatom x 3
            frac = tf.expand_dims(frac, axis=2) # batch x maxatom x 1 x 3
            fract = tf.transpose(frac, perm=[0, 2, 1, 3]) # batch x 1 x maxatom x 3
            sc = tf.reshape(self.get_sc(), shape=[1, -1, 1, 1, 3]) #  1 x nsc**3 x 1 x 1 x 3 
            dfrac = tf.expand_dims(fract - frac , axis = 1) # batch x 1 x maxatom x maxatom x 3
            dfrac = dfrac + sc # batch x nsc**3  x maxatom x maxatom x 3
            dreal = tf.einsum("blnmi,bij->blnmj", dfrac, latts) # batch x nsc**3  x maxatom x maxatom x 3
            dreal = tf.reduce_sum(tf.square(dreal), axis=-1, keepdims=True) # batch x nsc**3  x maxatom x maxatom x 1
            dreal = tf.where(tf.equal(dreal, 0), 9999*tf.ones_like(dreal), dreal) # maske out self image
            dreal = tf.sqrt(tf.reduce_min(dreal, axis=1)) # batch x maxatom x maxatom x 1
            #dreal = tf.reduce_min(dreal, axis=1) # batch x maxatom x maxatom x 1
            
            weight = tf.reduce_sum(coeff*tf.exp(-tf.abs(zeta)*dreal), axis=-2) # batch x maxatom x nzeta
            feat = tf.concat([atom_embeds, weight], axis=-1) 

        return feat


    def energy_func(self, atom_ids, coords, latts):

        with tf.variable_scope("attention"):

            #atom_embeds = self.get_atom_embed(atom_ids)
            #pos_embeds = self.get_pos_embed(latts, coords)
            #embeds = tf.concat((atom_embeds, pos_embeds), axis=-1)
            #embeds = self.stru_factor(atom_ids, latts, coords)
            embeds = self.dist_weight(atom_ids, latts, coords)
            #print(embeds)
            #import sys; sys.exit()

            masks1 = tf.cast(tf.not_equal(atom_ids, 0), tf.float32)
            masks2 = tf.equal(atom_ids, 0)

            layer1 = self.trans(embeds, masks2, "1")
            #layer1 = self.trans(layer1, masks2, "2")
            #layer3 = self.trans(layer2, masks2, "3")

            layer2 = tf.keras.layers.Dense(self.n_ff, activation=tf.nn.relu)(layer1)
            layer3 = tf.keras.layers.Dense(self.n_ff, activation=tf.nn.relu)(layer2)
            layer4 = tf.keras.layers.Dense(self.n_ff, activation=tf.nn.relu)(layer3)
            layer5 = tf.keras.layers.Dense(self.n_ff, activation=tf.nn.relu)(layer4)
            layer6 = tf.keras.layers.Dense(self.n_ff, activation=tf.nn.relu)(layer5)
            layer7 = tf.keras.layers.Dense(self.n_ff, activation=tf.nn.relu)(layer6)
            layer8 = tf.squeeze(tf.keras.layers.Dense(1, activation=None)(layer7))
            atomic_energy = masks1*layer8
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

            loss = tf.losses.huber_loss(energy/na, energy_p/na, delta=0.05) + \
                   tf.losses.huber_loss(force, force_p, delta=0.5) + \
                   tf.losses.huber_loss(stress, stress_p, delta=0.05)

            #loss = tf.losses.mean_squared_error(energy/na, energy_p/na) + \
            #       tf.losses.mean_squared_error(force, force_p) + \
            #       tf.losses.mean_squared_error(stress, stress_p)

            #loss = tf.losses.mean_squared_error(energy, energy_p) + \
            #       tf.losses.mean_squared_error(force, force_p) + \
            #       tf.losses.mean_squared_error(stress, stress_p)

            # Add code to check the broken structure
            self.check_ops = {}
            #cdt = tf.greater_equal(loss, 30)
            #self.check_ops.append(tf.cond(cdt, lambda: tf.print(energy_p/na), lambda: False))
            #self.check_ops.append(tf.cond(cdt, lambda: tf.print(energy/na), lambda: False))
            self.check_ops["energy_p"] = energy_p
            self.check_ops["energy"] = energy
            self.check_ops["force_p"] = force_p
            self.check_ops["force"] = force
            self.check_ops["atom_ids"] = atom_ids
            self.check_ops["latts"] = latts
            self.check_ops["coords"] = coords

            energy_loss_t = tf.sqrt(tf.reduce_mean(((energy-energy_p)/na)**2))
            #na = tf.reshape(na, (-1, 1, 1))
            force_loss_t = tf.sqrt(tf.reduce_mean(tf.reduce_sum(((force-force_p))**2, axis=-1)))
            stress_loss_t = tf.sqrt(tf.reduce_mean(((stress-stress_p))**2))

        return loss, energy_loss_t, force_loss_t, stress_loss_t

    def validate(self, atom_ids, coords, latts, force, energy, stress):

        with self.valid_graph.as_default():
            energy_p = self.energy_func(atom_ids, coords, latts)
            
            masks1 = tf.cast(tf.not_equal(atom_ids, 0), tf.float32)
            masks1 = tf.expand_dims(masks1, axis=-1)

            force_p = -tf.gradients(energy_p, coords)[0]*masks1
            vols = tf.reshape(tf.abs(tf.linalg.det(latts)), (-1, 1, 1))
            stress_p = -tf.einsum("bij,bkj->bik", tf.gradients(energy_p, latts)[0], latts)/vols

            masks = tf.cast(tf.not_equal(atom_ids, 0), tf.float32)
            na = tf.reduce_sum(masks, axis=1)
            self.avg_na =  tf.reduce_mean(na)

            loss = tf.losses.huber_loss(energy/na, energy_p/na, delta=0.05) + \
                   tf.losses.huber_loss(force, force_p, delta=0.5) + \
                   tf.losses.huber_loss(stress, stress_p, delta=0.05)

            energy_loss_t = tf.sqrt(tf.reduce_mean(((energy-energy_p)/na)**2))
            #na = tf.reshape(na, (-1, 1, 1))
            force_loss_t = tf.sqrt(tf.reduce_mean(tf.reduce_sum(((force-force_p))**2, axis=-1)))
            stress_loss_t = tf.sqrt(tf.reduce_mean(((stress-stress_p))**2))

        return loss, energy_loss_t, force_loss_t, stress_loss_t

    def infer(self, atom_ids, coords, latts):
        
        with self.infer_graph.as_default():
            energy_p = self.energy_func(atom_ids, coords, latts)
            
            masks1 = tf.cast(tf.not_equal(atom_ids, 0), tf.float32)
            masks1 = tf.expand_dims(masks1, axis=-1)

            force_p = -tf.gradients(energy_p, coords)[0]*masks1
            vols = tf.reshape(tf.abs(tf.linalg.det(latts)), (-1, 1, 1))
            stress_p = -tf.einsum("bij,bkj->bik", tf.gradients(energy_p, latts)[0], latts)/vols

        return energy_p, force_p, stress_p

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
