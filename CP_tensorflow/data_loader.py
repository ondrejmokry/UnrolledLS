import tensorflow as tf
import tensorflow_mri as tfmri
import scipy.io as sio
import matplotlib.pyplot as plt
import os
import math
import numpy as np

class MRI_dataset():
    def __init__(self, subfold="train"):
        # read the location of the folder with data
        f = open("../datafold.txt", "r")
        df = f.read().rstrip("\n")
        f.close()
        
        # set the subfolder structure
        self.df = df
        self.simfold = self.df + "/simulation/"
        self.echofold = self.df + "/" + subfold + "/" 
                
        # set the echo files
        self.files = os.listdir(self.echofold)
                    
    def __len__(self):
        return len(self.files)

    # load data and reference
    def load_item(self, idx):
                
        # echoes
        mat = sio.loadmat(self.echofold + self.files[idx])
        
        d=tf.cast(tf.convert_to_tensor(np.ascontiguousarray(mat['EchoSignals'])),dtype=tf.complex64)
        imgs=tf.convert_to_tensor(np.ascontiguousarray(mat['imgs']))
       # Angles=tf.convert_to_tensor(mat['Angles'])
        Sensitivities=tf.convert_to_tensor(np.ascontiguousarray(mat['Sensitivities']))
        # read the rest of the parameters
        # Nf = imgs.shape[2]
        Nf = 200
        rpf = int(mat["rpf"])
        factor = int(mat["factor"])
        if len(d.shape)<3:
            d=tf.expand_dims(d,axis=2)
        # crop the sequences
        #Angles= Angles[0,:Nf*rpf]
        d = d[:, :Nf*rpf, :] 
       # Angles=tf.reshape(Angles,(Nf,int(len(Angles)/Nf)))
        d=tf.transpose(d,perm=[2,1,0])

        d=tf.reshape(d,(d.shape[0],Nf,rpf*d.shape[2]))

        imgs = imgs[:, :, :Nf] 
        imgs=tf.transpose(imgs,perm=[2,0,1])
        Sensitivities=tf.cast(Sensitivities,dtype=tf.complex64)

        #add batch dim

        d=tf.expand_dims(d,axis=0)
        imgs=tf.expand_dims(imgs,axis=0)


        return Sensitivities, d, Nf, rpf, imgs, factor

    def load_data(self,batch_size=1,cartesian_reco=False):
        
        for i in range (self.__len__()):
            Sensitivities,d,Nf,rpf,imgs,factor=self.load_item(i)
            if(i==0):
                source=d
                gt=imgs
            else:
                source=tf.concat([source,d],axis=0)
                gt=tf.concat([gt,d],axis=0)

        kspace=tf.reverse(tfmri.radial_trajectory(base_resolution=64,views=rpf,phases=Nf,ordering='golden_half',angle_range='full'),[3]) # this should be ok, it is rotated 
        w=tfmri.sampling.flatten_density(tfmri.sampling.estimate_radial_density(kspace)) # is exact for radial trajectories
        kspace=tfmri.sampling.flatten_trajectory(kspace)

        #w = tfmri.sampling.estimate_density(kspace,(64,64),method="pipe")

        w=tf.cast(1/w,tf.complex64) # compensation is inverse of density
        
        

        if cartesian_reco:  # Convert to cartesian dataset
            
            source=tf.math.conj(Sensitivities)*tfmri.signal.nufft(source*w,grid_shape= (64,64),
                            points=kspace,transform_type='type_1',fft_direction='backward')
            source=tfmri.signal.fft(source,axes=(3,4),norm="ortho")

        dataset=tf.data.Dataset.from_tensor_slices((source,gt))
        dataset=dataset.batch(batch_size)

        return dataset, kspace,w
