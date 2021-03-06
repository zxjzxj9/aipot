#! /usr/bin/env python

"""
    This script conver the training data into TFRecord version
    to speedup data reading
   
    author: Victor (Xiao-Jie) Zhang
    date: 2018/05/26
"""

import tensorflow as tf
import numpy as np
import random
import yaml
from collections import Iterable
import random

class StrReader(object):
    def __init__(self, config="./config.yml"):
        self.config = yaml.load(open(config))
        self.str_file = self.config["str_file"]
        self.force_file = self.config["force_file"]
        #self.mapping = {
        #    self.config["atoms"][k]:self.config["embeds"][k] \
        #    for k in self.config["atoms"].keys() }

    def _parse_str(self, s):
        if not s.readline(): raise StopIteration
        energy = float(s.readline().split()[2])
        na = int(s.readline().split()[-1])
        s.readline() # element in structure
        s.readline() # symbol
        s.readline() # No.
        s.readline() # number
        vec1 = np.array(list(map(float, s.readline().split()[1:])))
        vec2 = np.array(list(map(float, s.readline().split()[1:])))
        vec3 = np.array(list(map(float, s.readline().split()[1:])))

        atoms = []
        coords = []
        charges = []
        for _ in range(na):
            tmp = s.readline().split()
            atom = int(tmp[1])
            atoms.append(atom)
            coords.append(list(map(float, tmp[2:5])))
            charge = float(tmp[2])

        s.readline() # End one structure
        s.readline() # blank line

        coords = np.stack(coords, axis=0)

        if na > self.config["max_atom_num"]:
            return None

        #print(np.pad(np.array(atoms), (0, self.config["max_atom_num"]-na), 'constant'))
        #print(self.mapping)
        #atoms = [self.mapping[at] for at in atoms]
        return na, energy, np.pad(np.array(atoms), (0, self.config["max_atom_num"]-na), 'constant'), \
               np.stack([vec1, vec2, vec3], axis=0),\
               np.pad(coords, ((0, self.config["max_atom_num"]-na), (0, 0)),\
                      'constant')

    def _parse_force(self, f, na):
        f.readline()
        sxx, sxy, sxz, syy, syz, szz = list(map(float, f.readline().split()[1:]))
        forces = []
        atoms = []
        for _ in range(na):
            tmp = f.readline().split()
            atom = int(tmp[1])
            atoms.append(atom)
            forces.append(list(map(float, tmp[2:])))
        #print(atoms)
        #print(self.mapping)
        #atoms = [self.mapping[at] for at in atoms]
        f.readline()
        f.readline()

        if na > self.config["max_atom_num"]:
            return None

        stress = np.array([[sxx, sxy, sxz], [sxy, syy, syz], [sxz, syz, szz]])
        return na, stress, np.array(atoms), np.pad(np.stack(forces, axis=0), \
                   ((0, self.config["max_atom_num"]-na), (0, 0)), 'constant')

    def __iter__(self):
        import functools
        with open(self.str_file) as s, \
             open(self.force_file) as f:
            while True:
                try:
                    tmp_str = self._parse_str(s)
                    tmp_for = self._parse_force(f, tmp_str[0])
                    if not tmp_str: continue 
                    assert functools.reduce(lambda x,y : x and y, zip(tmp_str[2], tmp_for[2]))
                    yield tmp_str, tmp_for
                except StopIteration:
                    return

class RecordWriter(object):
    def __init__(self, config="./config.yml"):
        self.config = yaml.load(open(config))
        self.train_writer = tf.python_io.TFRecordWriter(self.config["train_file"]) 
        self.val_writer = tf.python_io.TFRecordWriter(self.config["valid_file"])
        self.test_writer = tf.python_io.TFRecordWriter(self.config["test_file"])

    def _int64_feature(self, value):
        if not isinstance(value, Iterable):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
        elif isinstance(value, np.ndarray):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=value.flatten()))

    def _float_feature(self, value):
        if not isinstance(value, Iterable):
            return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
        elif isinstance(value, np.ndarray):
            return tf.train.Feature(float_list=tf.train.FloatList(value=value.flatten()))
    
    def write_feature(self, tmp_str, tmp_for, writer):
        feature = {\
            "na": self._int64_feature(tmp_str[0]),\
            "energy": self._float_feature(tmp_str[1]),\
            "serial": self._int64_feature(tmp_str[2]),\
            "latt": self._float_feature(tmp_str[3]),\
            "coord": self._float_feature(tmp_str[4]),\
            "stress": self._float_feature(tmp_for[1]),\
            "force": self._float_feature(tmp_for[3]) }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())
    
    def write(self, reader, aug_num=0):
        cnt = 0
        ntrain = 0
        nval = 0
        ntest = 0
      
        e_rec = []  
        
        for tmp_str, tmp_for in reader:
            if tmp_str[1] > self.config["max_energy"]: continue
            e_rec.append(tmp_str[1]/tmp_str[0])
            cnt += 1
            print("Current Num: {}/250k, Train Num: {}, Test Num: {}, Val Num: {}"\
                    .format(cnt, ntrain, nval, ntest), end="\r")
            prob = random.random()
            if prob < self.config["train_prob"]:
                ntrain += 1
                #print(tmp_str[3]); import sys;sys.exit() 

                if np.linalg.cond(tmp_str[3]) > 10**4:
                    print("ill condition cell matrix, skip...")
                    continue

                self.write_feature(tmp_str, tmp_for, self.train_writer)

                for i in range(aug_num):
                    l1, l2, l3 = random.random(), random.random(), random.random()
                    tmp_str[4][:tmp_str[0],:] += \
                        l1*tmp_str[3][0, :] + l2*tmp_str[3][1, :] + l3*tmp_str[3][2, :]
                    self.write_feature(tmp_str, tmp_for, self.train_writer)
                    
            elif prob > self.config["train_prob"] and prob < 1.0 - self.config["test_prob"]:
                nval += 1
                self.write_feature(tmp_str, tmp_for, self.val_writer)
            else:
                ntest += 1
                self.write_feature(tmp_str, tmp_for, self.test_writer)
        print("")
        print("Mean value of data: {:.2f}, std value of data {:.2f}".format(np.mean(e_rec), np.std(e_rec)))
        self.train_writer.close()
        self.val_writer.close()
        self.test_writer.close()
        
def create_db():
    sr = StrReader()
    rw = RecordWriter()
    rw.write(sr, aug_num=0)

if __name__ == "__main__":
    create_db()
