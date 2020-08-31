import argparse
import os
import numpy as np
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from datasets import FaceDataset
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch
import pdb


import torchvision.datasets as datasets
import torchvision
import torch
import gzip
from torchvision import datasets, transforms
import PIL
from PIL import Image
import torchvision.transforms.functional as TF

import os
from os.path import isdir, exists, abspath, join

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
from mpl_toolkits.mplot3d import Axes3D

import os
from os.path import isdir, exists, abspath, join

from dcgan import Generator, Discriminator

def main():
    
    pwd=os.getcwd()

    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    print (pwd)

    model = torch.load('/home/moorthy/skeleton/voxel/models/gen_18400.pt')

    print ("Model loaded")

    z1 = Variable( Tensor( np.random.normal( 0, 1, ( 64, 100 ) ) ) )

    z2 = Variable( Tensor( np.random.normal( 0, 1, ( 64, 100 ) ) ) )

    z3 = z2 - z1
    
    #print ('Z2 shape : ', z2.shape)

    #print ('Z1 Shape : ', z1.shape)

    interval = 6
    
    new_z = np.empty((64,100))


    for i in range(20):

        new_z = ((z1) + (z3/interval))

        interval = interval - 0.3
        
        #print ('ggbgbg', new_z.shape)

        new_z1 = model(new_z)     
       
        #print ('dwdvwv', new_z1.shape)

        new_z1 = new_z1 [0,:,:,:,:]

        new_z1 = new_z1.squeeze(0)

        #print ('New Z1 : ', new_z1.shape)

        interp_z = new_z1.cpu().detach().numpy()

        interp_z = interp_z > 0.2
        
        if i%3==0:

            fig = plt.figure()

            ax = fig.gca(projection='3d')

            ax.set_aspect('equal')

            ax.voxels(interp_z, edgecolor="k")

            ax.view_init(elev = 60)

            plt.savefig( str(i) + "--" +"interp_result.png")

if __name__ == '__main__':
    main()
