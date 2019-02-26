#! /usr/bin/env python

import tensorflow as tf
import yaml
from option import opts
import itertools
import numpy as np

class DensityNet(object):
    """
        Prototype of neural network to use 3D convolution
        max_atom: max atom number
        embed_size: atomic embedding size
        coeff_size: number of zetas
        nsc: supercell extenstion in x, y, z direction
        nmesh: 3d lattice mesh in x, y, z direction
    """
    def __init__(self, max_atom, embed_size, coeff_size, nsc, mean=0.0, std=1.0, nmesh=16):
        self.atom_kinds = 120 # index zero is unkown atoms
        self.max_atom = max_atom
        self.embed_size = embed_size
        self.coff_size = coeff_size
        self.nsc = nsc
        self.nmesh = nmesh
        self.test_ops = []
        self.mean = mean
        self.std = std
        self._build_graph()

    def get_embedding(self):
        with tf.variable_scope("atomic_embedding"):
            embed=tf.get_variable("embed", shape=(self.atom_kinds, self.embed_size),
                dtype=tf.float32, initializer=tf.initializers.random_normal(stddev=1.0e-3))
        return embed

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

    def get_mesh(self):
        """
            get lattice mesh, dim (nmesh**3)*3
        """
        mesh = []
        step = 1.0/self.nmesh
        a = np.array([step, 0, 0], dtype=np.float32)
        b = np.array([0, step, 0], dtype=np.float32)
        c = np.array([0, 0, step], dtype=np.float32)
        for i, j, k in itertools.product(*itertools.repeat(range(self.nmesh), 3)):
            mesh.append(i*a + j*b + k*c)
        mesh = np.stack(mesh, axis=0)

        with tf.variable_scope("latt_mesh"):
            mesh = tf.constant(mesh, dtype=tf.float32)
        return mesh

    def density_func(self, atom_ids, coords, latts, embed, masks):
        """
            Calculate the electron density give inputs
            density is given by sum(coeff*exp(-zeta*r))
            coords: in x y z, real space
            latt: real space
        """
        with tf.variable_scope("atomic_density", initializer=tf.keras.initializers.he_normal()):
            embeds = tf.nn.embedding_lookup(embed, atom_ids)
            expand_last = lambda x: tf.expand_dims(x, -1)
            expand_last2 = lambda x: expand_last(expand_last(x))

            with tf.variable_scope("density_param", initializer=tf.keras.initializers.he_normal()):
                zeta = tf.get_variable("zeta", shape=(self.coff_size, self.embed_size),
                    dtype=tf.float32, initializer=tf.initializers.random_uniform(minval=0.0, maxval=1.0))
                coeff = tf.get_variable("coeff", shape=(self.coff_size, self.embed_size),
                    dtype=tf.float32, initializer=tf.initializers.random_uniform(minval=0.0, maxval=1.0))

            with tf.variable_scope("density_calc", initializer=tf.keras.initializers.he_normal()):
                # embeds: batch x max_atom x embed_size
                # zeta/coeff: coff_size x embed_size
                # zetas/coeffs: batch x max_atom x coff_size
                # see https://stackoverflow.com/questions/38235555/tensorflow-matmul-of-input-matrix-with-batch-data/38244353
                zetas = expand_last2(tf.einsum("bij,kj->bik", embeds, zeta)) # batch x max_atom x coff_size x 1 x 1
                coeffs = expand_last2(tf.einsum("bij,kj->bik", embeds, coeff))

                sc = tf.expand_dims(self.get_sc(), axis=0) #  1 x nsc**3 x 3
                mesh = tf.expand_dims(self.get_mesh(), axis=1) # nmesh**3 x 1 x 3

                # Notice Row Major or Column Major !!!
                inv_latts = tf.matrix_inverse(latts)
                frac = tf.mod(tf.einsum("bij,bjk->bik", coords, inv_latts), 1.0) # batch x maxatom x 3
                frac = tf.expand_dims(frac, axis=2) # batch x maxatom x 1 x 3
                frac_all = tf.expand_dims(frac + sc, axis=2) # batch x maxatom x 1 x nsc**3  x 3
                frac_dvec = frac_all - mesh # batch x maxatom x nmesh**3 x nsc**3 x 3
                real_dvec = tf.einsum("bijkm,bmn->bijkn", frac_dvec, latts) # batch x maxatom x nsc**3 x nmesh**3 x 3
                dist_sq = tf.reduce_sum(tf.square(real_dvec), axis=-1) # batch x maxatom x nsc**3 x nmesh**3
                # self.test_op = tf.print(tf.reduce_min(dist_sq))
                # Get max contribution atom location
                dist_sq = tf.sqrt(tf.maximum(dist_sq, 1e-6))
                # dist = tf.reduce_min(tf.sqrt(dist_sq, name="dist"), axis=2)  # batch x maxatom x nmesh**3
                dist = -tf.nn.top_k(-dist_sq, k=3, sorted=False)[0]  # batch x maxatom x nmesh**3 x 6
                dist = tf.expand_dims(dist, 2, name="expand_dist") # batch x maxatom x 1 x nmesh**3 x 6
                den1 = coeffs*tf.exp(-tf.abs(zetas*dist)) # batch x maxatom x coff_size x nmesh**3 x 6
                den1 = tf.reshape(masks, (-1, self.max_atom, 1, 1, 1))*den1 # mask atoms
                den_tot = tf.reduce_sum(den1, [1, 2, 4]) # batch x nmesh**3
                #print(den_tot); import sys; sys.exit()
                # print(den_tot)
                # import sys; sys.exit()
                # Reshape and gives us the final charge density
                den_tot = tf.reshape(den_tot, [-1, self.nmesh, self.nmesh, self.nmesh, 1]) # batch x self.nmesh x self.nmesh x self.nmesh x 1
        return den_tot

    def residue_conn(self, tensor, channel_out, channel_middle, scope, training=True):
        with tf.variable_scope(scope, initializer=tf.keras.initializers.he_normal()):
            layer2 = tf.layers.conv3d(tensor, channel_out, (3,3,3), (2,2,2), padding='same', activation=None)
            layer2 = tf.layers.batch_normalization(layer2, training=training)
            layer2 = tf.nn.relu(layer2)
            layer3 = tf.layers.conv3d(layer2, channel_middle, (3,3,3), (1,1,1), padding='same', activation=None)
            layer3 = tf.layers.batch_normalization(layer3, training=training)
            layer2 = tf.nn.relu(layer3)
            layer4 = tf.layers.conv3d(layer3, channel_out, (3,3,3), (1,1,1), padding='same', activation=None)
            layer4 = tf.layers.batch_normalization(layer3, training=training)
            layer4 = tf.nn.relu(layer2+layer4)
        return layer4

    def energy_func(self, density, training=True):

        with tf.variable_scope("energy_conv", initializer=tf.keras.initializers.he_normal()):
            layer1 = tf.layers.conv3d(density, 16, (3,3,3), (1,1,1), padding='same', activation=tf.nn.relu)
            ## 32->16, 16->8, 8->4, 4->2, 2->1
            # 16->8, 8->4, 4->2, 2->1
            res1 = self.residue_conn(layer1, 32, 64, 'res1', training)
            res2 = self.residue_conn(res1, 64, 128, 'res2', training)
            res3 = self.residue_conn(res2, 128, 256, 'res3', training)
            res4 = self.residue_conn(res3, 256, 512, 'res4', training)
            res5 = res4
            #res5 = self.residue_conn(res4, 512, 1024, 'res5', training) # nbatch x 1 x 1 x 1 x 512
            layer6 = tf.layers.conv3d(res5, 1, (1,1,1), (1,1,1), padding='same')
            #print(layer6)
            energy = self.std*tf.squeeze(layer6) + self.mean
            #std = tf.Variable(1, dtype=tf.float32)
            #mean = tf.Variable(0, dtype=tf.float32)
            #energy = std*tf.exp(tf.squeeze(layer6)) + mean
        return energy

    def _build_graph(self):
        self.train_graph = tf.Graph()
        self.valid_graph = tf.Graph()
        self.infer_graph = tf.Graph()

    def train(self, atom_ids, coords, latts, forces, energies, stresses):
        with self.train_graph.as_default():
            embed = self.get_embedding()
            masks = tf.cast(tf.not_equal(atom_ids, 0), tf.float32)
            density = self.density_func(atom_ids, coords, latts, embed, masks)
            energies_p = self.energy_func(density, True)
            forces_p = -tf.gradients(energies_p, coords)[0]
            vols = tf.reshape(tf.abs(tf.linalg.det(latts)), (-1, 1, 1))
            #inv_latts = tf.matrix_inverse(latts)
            stresses_p = -tf.einsum("bij,bkj->bik", tf.gradients(energies_p, latts)[0], latts)/vols
            na = tf.reduce_sum(masks, axis=1)
            # loss = energies_p
            # self.test_ops.append(tf.print(energies))
            # self.test_ops.append(tf.print(energies_p))
            #delta_energy_per_atom = (energies-energies_p)/na
            #delta_force = forces-forces_p
            #delta_stress = stresses-stresses_p
            loss = tf.losses.huber_loss(energies/na, energies_p/na) + \
                   tf.losses.huber_loss(forces, forces_p) + \
                   tf.losses.huber_loss(stresses, stresses_p)
            #loss = tf.reduce_mean((energies-energies_p)**2/na) \
            #    + tf.reduce_mean((forces-forces_p)**2) \
            #    + tf.reduce_mean((stresses-stresses_p)**2)
            #  loss = tf.reduce_mean((energies-energies_p)**2/na) 
            energy_loss_t = tf.sqrt(tf.reduce_mean((energies-energies_p)**2/na))
            force_loss_t = tf.sqrt(tf.reduce_mean((forces-forces_p)**2))
            stress_loss_t = tf.sqrt(tf.reduce_mean((stresses-stresses_p)**2))
        return loss, energy_loss_t, force_loss_t, stress_loss_t


    def validation(self, atom_ids, coords, latts, forces, energies, stresses):
        with self.train_graph.as_default():
            embed = self.get_embedding()
            masks = tf.cast(atom_ids != 0, tf.float32)
            density = self.density_func(atom_ids, coords, latts, embed, masks)
            energies_p = self.energy_func(density, False)
            forces_p = tf.gradients(energy, coords)[0]
            vols = tf.reshape(tf.abs(tf.linalg.det(latts)), (-1, 1, 1))
            #inv_latts = tf.matrix_inverse(latts)
            stresses_p = -tf.einsum("bij,bkj->bik", tf.gradients(energies_p, latts)[0], latts)/vols

            #loss = 0.1*tf.reduce_sum((energies-energies_p)**2)/tf.reduce_sum(masks)**2 \
            #    + tf.reduce_mean((forces-forces_p)**2) \
            #    + tf.reduce_mean((stresses-stresses_p)**2)

            energy_loss_t = tf.sqrt(tf.reduce_sum((energies-energies_p)**2)/tf.reduce_sum(masks))
            force_loss_t = tf.sqrt(tf.reduce_mean((forces-forces_p)**2))
            stress_loss_t = tf.sqrt(tf.reduce_mean((stresses-stresses_p)**2))
            loss = energy_loss_t + force_loss_t + stress_loss_t

        return loss, energy_loss_t, force_loss_t, stress_loss_t


    def inference(self, atom_ids, coords, latts):
        with self.infer_graph.as_default():
            #atom_ids = tf.placeholder(tf.int32, shape=(None, self.max_atom))
            #masks = tf.placeholder(tf.int32, shape=(None, self.max_atom))
            #coords = tf.placeholder(tf.float32, shape=(None, self.max_atom, 3))
            #latts = tf.placeholder(tf.float32, shape=(None, 3, 3))
            embed = self.get_embedding()
            masks = tf.cast(atom_ids != 0, tf.float32)
            density = self.density_func(atom_ids, coords, latts, embed, masks)
            energy_p = self.energy_func(density, False)
            force_p = tf.gradients(self.energy_p, self.coords)[0]
            vols = tf.reshape(tf.abs(tf.linalg.det(latts)), (-1, 1, 1))
            inv_latts = tf.matrix_inverse(latts)
            stress_p = -tf.einsum("bij,bkj->bik", tf.gradients(energy_p, latts)[0], inv_latts)*vols
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
