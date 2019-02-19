#! /usr/bin/env python

import tensorflow as tf
import yaml
from option import opts
import itertools

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

class DensityNet(object):
    """
        Prototype of neural network to use 3D convolution
        max_atom: max atom number
        embed_size: atomic embedding size
        coeff_size: number of zetas
        nsc: supercell extenstion in x, y, z direction
        nmesh: 3d lattice mesh in x, y, z direction
    """
    def __init__(self, max_atom, embed_size, coeff_size, nsc, nmesh=32):
        self.atom_kinds = 120 # index zero is unkown atoms
        self.max_atom = max_atom
        self.embed_size = embed_size
        self.coff_size = coeff_size
        self.nsc = nsc
        self.nmesh = nmesh

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
        for i, j, k in itertools.product(*itertools.repeat(range(-self.nsc//2,self.nsc//2+1), 3)):
            sc.append(i*a + j*b + k*c)
        sc = np.concatenate(sc, axis=0)

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
        mesh = np.concatenate(mesh, axis=0)

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
        with tf.variable_scope("atomic_density"):
            embeds = tf.nn.embedding_lookup(embed, atom_ids)
            expand_last = lambda x: tf.expand_dims(x, -1)
            expand_last2 = lambda x: expand_last(expand_last(x))

            with tf.variable_scope("density_param"):
                zeta = tf.get_variable("zeta", shape=(self.coff_size, self.embed_size),
                    dtype=tf.float32, initializer=tf.initializers.random_uniform(maxval=1.0))
                coeff = tf.get_variable("coeff", shape=(self.coff_size, self.embed_size),
                    dtype=tf.float32, initializer=tf.initializers.random_uniform(maxval=1.0))

            with tf.variable_scope("density_calc"):
                # embeds: batch x max_atom x embed_size
                # zeta/coeff: coff_size x embed_size
                # zetas/coeffs: batch x max_atom x coff_size
                # see https://stackoverflow.com/questions/38235555/tensorflow-matmul-of-input-matrix-with-batch-data/38244353
                zetas = expand_last2(tf.einsum("bij,kj->bik", embeds, zeta)) # batch x max_atom x coff_size x 1 x 1
                coeffs = expand_last2(tf.einsum("bij,kj->bik", embeds, coeff))

                sc = tf.expand_dims(self.get_sc(), axis=0) # 1 x nsc**3 x 3
                mesh = tf.expand_dims(self.get_mesh(), axis=0) # 1 x nmesh**3 x 3

                # Notice Row Major or Column Major !!!
                inv_latts = tf.matrix_inverse(latts)
                frac = tf.einsum("bij,bjk->bik", inv_latts, coords) # batch x maxatom x 3
                frac = tf.expand_dims(frac, axis=2) # batch x maxatom x 1 x 3
                frac_all = tf.expand_dims(frac + sc, axis=3) # batch x maxatom x nsc**3 x 1 x 3
                frac_dvec = frac_all - mesh # batch x maxatom x nsc**3 x nmesh**3 x 3
                real_dvec = tf.einsum("bijkm,bmn->bijkn", frac_dvec, latts) # batch x maxatom x nsc**3 x nmesh**3 x 3
                dist = tf.expand_dims(tf.sqrt(tf.sum(real_dvec**2, dim=-1)), 2) # batch x maxatom x 1 x nsc**3 x nmesh**3
                den1 = coeff*tf.exp(-zeta*dist) # batch x maxatom x coff_size x nsc**3 x nmesh**3
                den1 = tf.cast(masks, tf.float32)*den1 # mask atoms
                den_tot = tf.reduce_sum(den1, [1, 2, 3]) # batch x nmesh**3
                # Reshape and gives us the final charge density
                den_tot = tf.reshape(den_tot, [-1, self.nmesh, self.nmesh, self.nmesh, 1]) # batch x self.nmesh x self.nmesh x self.nmesh x 1
        return den_tot

    def residue_conn(self, tensor, channel_out, scope, training=True):
        with tf.varaible_scope(scope):
            layer2 = tf.layers.conv2d(tensor, channel_out, (3,3,3), (2,2,2), padding='same', activation=None)
            layer2 = tf.layers.batch_normalization(layer2, activation_fn=tf.nn.relu, training=training)
            layer3 = tf.layers.conv2d(layer2, 32, (3,3,3), (1,1,1), padding='same', activation=None)
            layer3 = tf.layers.batch_normalization(layer3, activation_fn=tf.nn.relu, training=training)
            layer4 = tf.layers.conv2d(layer2, 32, (3,3,3), (1,1,1), padding='same', activation=None)
            layer4 = tf.layers.batch_normalization(layer3, activation_fn=None, training=training)
            layer4 = tf.nn.relu(layer4)
        return layer4

    def energy_func(self, density, training=True):

        with tf.variable_scope("energy_conv"):
            layer1 = tf.layers.conv2d(density, 16, (3,3,3), (1,1,1), padding='same', activation=tf.nn.relu)
            # 32->16, 16->8, 8->4, 4->2, 2->1
            res1 = residue_conn(layer1, 32, 'res1', training)
            res2 = residue_conn(res1, 64, 'res2', training)
            res3 = residue_conn(res2, 128, 'res3', training)
            res4 = residue_conn(res3, 256, 'res4', training)
            res5 = residue_conn(res4, 512, 'res5', training) # nbatch x 1 x 1 x 1 x 512

            layer6 = tf.layers.conv2d(res5, 1, (1,1,1), (1,1,1), padding='same', activation=tf.nn.relu)
            energy = tf.squeeze(layer6)
        return energy

    def _build_graph(self):
        self.train_graph = tf.Graph()
        self.valid_graph = tf.Graph()
        self.infer_graph = tf.Graph()

    def train(self, atom_ids, masks, coords, latts, forces, energies, stresses):
        with self.train_graph.as_default():
            embed = self.get_embedding()
            density = self.density_func(atom_ids, coords, latts, embed, masks)
            energies_p = self.energy_func(density, True)
            forces_p = tf.gradients(energy, coords)
            vols = tf.abs(tf.linalg.det(latts))
            inv_latts = tf.inv_latts(latts)
            stresses_p = -tf.einsum("bij,bkj->bik", tf.gradients(energy, latts), inv_latts)*vols

            loss = tf.reduce_mean((energies-energies_p)**2) \
                + 1e-3*tf.reduce_mean((forces-forces_p)**2)
                + 1e-3*tf.reduce_mean((stresses-stresses_p)**2)

            energy_loss_t = tf.sqrt(tf.reduce_mean((energies-energies_p)**2))
            force_loss_t = tf.sqrt(tf.reduce_mean((forces-forces_p)**2))
            stress_loss_t = tf.sqrt(tf.reduce_mean((stresses-stresses_p)**2))
        return loss, energy_loss_t, force_loss_t, stress_loss_t


    def validation(self, atom_ids, masks, coords, latts, forces, energies, stresses):
        with self.train_graph.as_default():
            embed = self.get_embedding()
            density = self.density_func(atom_ids, coords, latts, embed, masks)
            energies_p = self.energy_func(density, False)
            forces_p = tf.gradients(energy, coords)
            vols = tf.abs(tf.linalg.det(latts))
            inv_latts = tf.inv_latts(latts)
            stresses_p = -tf.einsum("bij,bkj->bik", tf.gradients(energy, latts), inv_latts)*vols

            loss = tf.reduce_mean((energies-energies_p)**2) \
                + 1e-3*tf.reduce_mean((forces-forces_p)**2)
                + 1e-3*tf.reduce_mean((stresses-stresses_p)**2)

            energy_loss_t = tf.sqrt(tf.reduce_mean((energies-energies_p)**2))
            force_loss_t = tf.sqrt(tf.reduce_mean((forces-forces_p)**2))
            stress_loss_t = tf.sqrt(tf.reduce_mean((stresses-stresses_p)**2))
        return loss, energy_loss_t, force_loss_t, stress_loss_t


    def inference(self, atom_ids, masks, coords, latts):
        with self.infer_graph.as_default():
            #atom_ids = tf.placeholder(tf.int32, shape=(None, self.max_atom))
            #masks = tf.placeholder(tf.int32, shape=(None, self.max_atom))
            #coords = tf.placeholder(tf.float32, shape=(None, self.max_atom, 3))
            #latts = tf.placeholder(tf.float32, shape=(None, 3, 3))
            embed = self.get_embedding()
            density = self.density_func(atom_ids, coords, latts, embed, masks)
            energy_p = self.energy_func(density, False)
            force_p = tf.gradients(self.energy_p, self.coords)
            vols = tf.abs(tf.linalg.det(latts))
            stress_p = -tf.einsum("bij,bkj->bik", tf.gradients(energy, latts), inv_latts)*vols
        return energy_p, force_p, stress_p

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
            self.train_loss, self.iterator = self.eval_loss(self.config["train_file"])
        
        with self.val_graph.as_default():
            self.val_loss = self.eval_loss(self.config["val_file"])

        with self.infer_graph.as_default():
            self.na = tf.placeholder(dtype=tf.int64, shape=(None,))
            self.ntyp = tf.placeholder(dtype=tf.int64, shape=(None, self.config["max_atom_num"]))
            self.latt = tf.placeholder(dtype=tf.float32, shape=(None, 3, 3))
            self.coords = tf.placeholder(dtype=tf.float32, shape=(None, self.config["max_atom_num"], 3))
            self.pes = self.calc_pes(self.na, self.ntyp, self.latt, self.coords)
            
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
        return loss, iterator

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

        atom_embeds = tf.nn.embedding_lookup(embeds, ntyp)
        atom_input = tf.concat((atom_embeds, coords), axis = 1)

        rnn_func = lambda x: tf.nn.rnn_cell.BasicLSTMCell(num_units=opts.nunits)

        with tf.variable_scope("rnn", initializer=tf.contrib.layers.xavier_initializer()):
            cell_fw = tf.nn.rnn_cell.MultiRNNCell([rnn_func() for _ in range(opts.nlayers)])
            cell_bw = tf.nn.rnn_cell.MultiRNNCell([rnn_func() for _ in range(opts.nlayers)])
            outputs, outputs_states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, atom_input, \
                sequence_length=na, dtype=tf.float32)

        atom_feat = tf.concat(outputs_states, axis=1)
        latt_feat = tf.reshape(latt, (-1, 9))

        total_feat = tf.concat((atom_feat, latt_feat), axis=1)

        with tf.variable_scope("linear", initializer=tf.contrib.layers.xavier_initializer()):
            linear_out = tf.layers.dense(total_feat, opts.nfeats, activation=tf.nn.relu)
            feats = tf.reshape(linear_out, shape=(-1, 32, 32, 1))
            feats = tf.layers.conv2d(feats, 32, (1, 1), padding="same", activation=tf.nn.relu)


        with tf.variable_scope("conv", initializer=tf.contrib.layers.xavier_initializer()):
            # 32x32 -> 16x16
            res1 = self.residue_block(feats, 64, 32, "residue1")
            res1 = tf.layers.conv2d(res1, 32, (2, 2), (2, 2), padding="same", activation=tf.nn.relu)
            # 16x16 -> 8x8
            res2 = self.residue_block(res1, 128, 64, "residue2")
            res2 = tf.layers.conv2d(res2, 32, (2, 2), (2, 2), padding="same", activation=tf.nn.relu)
            # 8x8 -> 4x4
            res3 = self.residue_block(res2, 256, 128, "residue3")
            res3 = tf.layers.conv2d(res3, 32, (2, 2), (2, 2), padding="same", activation=tf.nn.relu)
            # 4x4 -> 2x2
            res4 = self.residue_block(res3, 512, 256, "residue4")
            res4 = tf.layers.conv2d(res4, 32, (2, 2), (2, 2), padding="same", activation=tf.nn.relu)
            # 2x2 -> 1x1
            res5 = self.residue_block(res4, 1024, 512, "residue4")
            res5 = tf.layers.conv2d(res5, 32, (2, 2), (2, 2), padding="same", activation=tf.nn.relu)

        resout = tf.reshape(res5, (-1, 512))


        with tf.variable_scope("output", initializer=tf.contrib.layers.xavier_initializer()):
            e_out = tf.layers.dense(resout, units=1)
            f_out = tf.gradients(e_out, coords)
            s_out = tf.gradients(e_out, latt)
            # transpose s_out to be batchsize x 3 x 3
            s_out = tf.reshape(s_out, (-1, 3, 3))
            s_out = tf.matmul(s_out, latt, transpose_b=True)

        return e_out, f_out, s_out

    def residue_block(self, inputs, mid_chan, out_chan, kernel_size=3, name=None):
        with tf.variable_scope(name=name, initializer=tf.contrib.layers.xavier_initializer()):
            input = tf.layers.conv2d(inputs, out_chan, 1, 1, "same", activation=tf.nn.relu)
            layer1 = tf.layers.conv2d(inputs, mid_chan, 3, 1, "same", activation=tf.nn.relu)
            layer2 = tf.layers.conv2d(layer1, out_chan, 3, 1, "same", activation=None)
            return tf.nn.relu(input + layer2)

    def train(self):
        """
            Traing the model
        """
        with tf.Session(graph=self.train_graph) as s:
            optimizer = tf.train.AdamOptimizer(learning_rate=1e-3, beta1=0.9, beta2=0.999)
            train_op = optimizer.minimize(self.train_loss)

            s.run(tf.global_variables_initializer())
            s.run(self.iterator.initializer())

            writer = tf.summary.FileWriter("./logs", self.train_graph)
            tf.summary.scalar("loss", self.train_loss)
            ms_op = tf.summary.merge_all()

            saver = tf.train.Saver()

            cnt = 0
            while True:
                cnt += 1
                try:
                    loss, _, ms = s.run([self.train_loss, train_op, ms_op], feed_dict={})
                    writer.add_summary(ms_op, cnt)

                    if cnt % 1000 == 0:
                        saver.save(s, "./models")

                except tf.errors.OutOfRangeError:
                    print("Fininshed training...")




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
