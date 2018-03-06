'''
Created on Dec 14, 2017

@author: fwolf
'''

import caffe
import lmdb
import numpy as np

import cv2

from caffe.proto import caffe_pb2


class FeatureTripletLayer(caffe.Layer):
    """
    Creates triplet of word images.
    """
    def setup(self, bottom, top):
        if len(top) != 3:
            raise Exception("Need three outputs.")
        
        self.counter = 0;
        params = eval(self.param_str)
        self.file  = params["source"]
        self.db = lmdb.open(params["source"], readonly=True)
        self.cursor = self.db.begin().cursor()
        
        self.n_entries = self.db.begin().stat()['entries']
        
        self.random_indices = list(xrange(1,self.n_entries+1))
        
        np.random.shuffle(self.random_indices)
        
        self.cursor.next()
        self.n_count = int(self.cursor.key().split('_')[1])

        
    def reshape(self, bottom, top):
        top[0].reshape(1,22080)
        top[1].reshape(1,22080)
        top[2].reshape(1,22080)
        
        


    def forward(self, bottom, top):
        # get next random image from word list
        p1 = self.random_indices[self.counter % len(self.random_indices)]
        
        class_id = (p1-1) / self.n_count
        
        # find random image from same class
        p2 = p1
        
        while p2 == p1:
            p2 = np.random.randint(class_id*self.n_count+1, class_id*self.n_count+self.n_count+1)
        
        # find random image from different class
        n = p1
        
        while (n-1)/self.n_count == class_id:
            n = np.random.randint(0,self.n_entries) 
        
        key_p1 = '%s_%i' % (str(p1).zfill(8), self.n_count)
        key_p2 = '%s_%i' % (str(p2).zfill(8), self.n_count)
        key_n = '%s_%i' % (str(n).zfill(8), self.n_count)
        
        
        
        # Output 
        
        # first image
        value = self.cursor.get(key_p1)

        datum = caffe_pb2.Datum()
        datum.ParseFromString(value)
        
        data0 = caffe.io.datum_to_array(datum).astype(np.float32)
        top[0].data[...] = data0
        
        # second image
        value = self.cursor.get(key_p2)

        datum = caffe_pb2.Datum()
        datum.ParseFromString(value)
        
        data1 = caffe.io.datum_to_array(datum).astype(np.float32)
        top[1].data[...] = data1
        
        # third image
        value = self.cursor.get(key_n)

        datum = caffe_pb2.Datum()
        datum.ParseFromString(value)
        
        data2 = caffe.io.datum_to_array(datum).astype(np.float32)
        top[2].data[...] = data2 
        
        
        self.counter = self.counter +1
   
        
    

    def backward(self, top, propagate_down, bottom):
        pass

        
        