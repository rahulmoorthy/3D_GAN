# Reference: 
# https://github.com/eriklindernoren/Keras-GAN
# https://github.com/znxlwm/pytorch-MNIST-CelebA-GAN-DCGAN
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


parser = argparse.ArgumentParser()
parser.add_argument( '--n_epochs',
                     type=int,
                     default=20,
                     help='number of epochs of training' )
parser.add_argument( '--batch_size',
                     type=int,
                     default=64,
                     help='size of the batches' )
parser.add_argument( '--lr',
                     type=float,
                     default=0.0004,
                     help='adam: learning rate' )
parser.add_argument( '--b1',
                     type=float,
                     default=0.5,
                     help='adam: decay of first order momentum of gradient' )
parser.add_argument( '--b2',
                     type=float,
                     default=0.999,
                     help='adam: decay of first order momentum of gradient' )
parser.add_argument( '--n_cpu',
                     type=int,
                     default=8,
                     help='number of cpu threads to use during batch generation' )
parser.add_argument( '--latent_dim',
                     type=int,
                     default=100,
                     help='dimensionality of the latent space' )
parser.add_argument( '--img_size',
                     type=int,
                     default=64,
                     help='size of each image dimension' )
parser.add_argument( '--channels',
                     type=int,
                     default=3,
                     help='number ofi image channels' )
parser.add_argument( '--sample_interval',
                     type=int,
                     default=400,
                     help='interval between image sampling' )
# These files are already on the VC server. Not sure if students have access to them yet.
parser.add_argument( '--train_csv',
                     type=str,
                     default='/home/csa102/gruvi/celebA/train.csv',
                     help='path to the training csv file' )
parser.add_argument( '--train_root',
                     type=str,
                     default='/home/csa102/gruvi/celebA',
                     help='path to the training root' )
opt = parser.parse_args()

class Generator( nn.Module ):
    def __init__( self, d=64 ):
        super( Generator, self ).__init__()
        self.deconv1 = nn.ConvTranspose3d( opt.latent_dim, d * 8, 4, 1, 1 )
        self.deconv1_bn = nn.BatchNorm3d( d * 8 )
        self.deconv2 = nn.ConvTranspose3d( d * 8, d * 4, 4, 2, 1 )
        self.deconv2_bn = nn.BatchNorm3d( d * 4 )
        self.deconv3 = nn.ConvTranspose3d( d * 4, d * 2, 4, 2, 1 )
        self.deconv3_bn = nn.BatchNorm3d( d * 2 )
        self.deconv4 = nn.ConvTranspose3d( d * 2, d, 4, 2, 1 )
        self.deconv4_bn = nn.BatchNorm3d( d )
        self.deconv5 = nn.ConvTranspose3d( d, 1, 4, 2, 1 )

    # weight_init
    def weight_init( self, mean, std ):
        for m in self._modules:
            normal_init( self._modules[ m ], mean, std )

    # forward method
    def forward( self, input ):
        # x = F.relu(self.deconv1(input))
        x = input.view( -1, 100, 1, 1, 1)
        x = F.relu( self.deconv1_bn( self.deconv1( x ) ) )
        x = F.relu( self.deconv2_bn( self.deconv2( x ) ) )
        x = F.relu( self.deconv3_bn( self.deconv3( x ) ) )
        x = F.relu( self.deconv4_bn( self.deconv4( x ) ) )
        x = F.tanh( self.deconv5( x ) )
        #print ('GEN X SHAPE : ', x.shape )
        return x

class Discriminator( nn.Module ):
    # initializers
    def __init__( self, d=64 ):
        
        super( Discriminator, self ).__init__()
        
        self.conv1 = nn.Conv3d( 1, d, 4, 2, 1 )
        self.conv2 = nn.Conv3d( d, d * 2, 4, 2, 1 )
        self.conv2_bn = nn.BatchNorm3d( d * 2 )
        self.conv3 = nn.Conv3d( d * 2, d * 4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm3d( d * 4 )
        self.conv4 = nn.Conv3d( d * 4, d * 8, 4, 2, 1 )
        self.conv4_bn = nn.BatchNorm3d( d * 8 )
        self.conv5 = nn.Conv3d( d * 8, 1, 4, 1, 1)

    # weight_init
    def weight_init( self, mean, std ):
        for m in self._modules:
            normal_init( self._modules[ m ], mean, std )

    # forward method
    def forward( self, input ):
        
        #print ('Input Shape', input.shape)
        #x = input.view( -1, 100, 1, 1, 1 )
        x = F.leaky_relu( self.conv1( input ), 0.2 )
        x = F.leaky_relu( self.conv2_bn( self.conv2( x ) ), 0.2 )
        x = F.leaky_relu( self.conv3_bn( self.conv3( x ) ), 0.2 )
        x = F.leaky_relu( self.conv4_bn( self.conv4( x ) ), 0.2 )
        x = F.sigmoid( self.conv5( x ) )

        #print ('X: ' ,x.shape)
        return x

def normal_init( m, mean, std ):
    if isinstance( m, nn.ConvTranspose3d ) or isinstance( m, nn.Conv3d ):
        m.weight.data.normal_( mean, std )
        m.bias.data.zero_()

def main():
    
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    torch.backends.cudnn.benchmark = True
    # Loss function
    adversarial_loss = torch.nn.BCELoss()
    # Initialize generator and discriminator
    generator = Generator()
    discriminator = Discriminator()
    if cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()
    # Initialize weights
    generator.weight_init( mean=0.0, std=0.02 )
    discriminator.weight_init( mean=0.0, std=0.02 )
    # Configure data loader

    train_loader_MNIST = torch.utils.data.DataLoader(
            datasets.MNIST('mnist_data', train=True, download=True,                    
            transform=transforms.Compose([
            transforms.Resize((32,32)),
            transforms.ToTensor()])),batch_size=64, num_workers=1, shuffle=True)

    #Optimizers
    
    optimizer_G = torch.optim.Adam( generator.parameters(),
                                    lr=opt.lr,
                                    betas=( opt.b1, opt.b2 ) )
    optimizer_D = torch.optim.Adam( discriminator.parameters(),
                                    lr=opt.lr,
                                    betas=( opt.b1, opt.b2 ) )
    # ----------
    #  Training
    # ----------
    os.makedirs( 'images', exist_ok=True )
    os.makedirs( 'models', exist_ok=True )
   
   #Implemented Logic here to convert and handle 3D MNIST data

    for epoch in range( opt.n_epochs ):
        # learning rate decay
        if ( epoch + 1 ) == 11:
            optimizer_G.param_groups[ 0 ][ 'lr' ] /= 10
            optimizer_D.param_groups[ 0 ][ 'lr' ] /= 10
            print( 'learning rate change!' )
        if ( epoch + 1 ) == 16:
            optimizer_G.param_groups[ 0 ][ 'lr' ] /= 10
            optimizer_D.param_groups[ 0 ][ 'lr' ] /= 10
            print( 'learning rate change!' )
        
        for i, ( imgs, _ ) in enumerate( train_loader_MNIST ):

            print('image tensor shape (N, C, H, W):', imgs.shape)
            
            #print ('Idx : ', i)

            result_list = []

            for j in range (imgs.shape[0]):
                
                new = np.dstack([imgs[j][0]]*20)

                result = np.zeros((32,32,32))

                result[:new.shape[0], :new.shape[1], :new.shape[2]] = new

                #print (result.shape)
        
                result_display = result

                result = torch.Tensor(result)
        
                #print ('Result display' , result_display.shape)

                result = result.unsqueeze(0)
        
                result = result.unsqueeze(0)
    
                #result = torch.cat((result, result), dim=0)
        
                if j == 0:
                    result_list = result
            
                else:
                                
                    result_list = torch.cat((result_list, result), dim=0)

                #print (result_list.shape)

            #pdb.set_trace()
            
            # Adversarial ground truths
            
            valid = Variable( Tensor( result_list.shape[ 0 ], 1 ).fill_( 1.0 ),
                              requires_grad=False )
            fake = Variable( Tensor( result_list.shape[ 0 ], 1 ).fill_( 0.0 ),
                             requires_grad=False )
            
            # Configure input
            
            real_imgs = Variable( result_list.type( Tensor ) )
            
            #print ('Real Img Shape : ', real_imgs.shape)
            
            
            # -----------------
            #  Train Generator
            # -----------------
            
            #print ('Optim G')

            optimizer_G.zero_grad()
            
            # Sample noise as generator input
            
            z = Variable( Tensor( np.random.normal( 0, 1, ( result_list.shape[ 0 ],
                                                            opt.latent_dim ) ) ) )
            
            # Generate a batch of images
            
            gen_imgs = generator( z )
            
            # Loss measures generator's ability to fool the discriminator
            
            g_loss = adversarial_loss( discriminator( gen_imgs ), valid )
            
            g_loss.backward()
            
            optimizer_G.step()
            
            # ---------------------
            #  Train Discriminator
            # ---------------------
            
            optimizer_D.zero_grad()
            
            #print ('Optim D')

            #pdb.set_trace()

            # Measure discriminator's ability to classify real from generated samples
            label_real = discriminator( real_imgs )
            #print ('Label Real Size : ', label_real.shape)
            label_gen = discriminator( gen_imgs.detach() )
            real_loss = adversarial_loss( label_real, valid )
            fake_loss = adversarial_loss( label_gen, fake )
            d_loss = ( real_loss + fake_loss ) / 2
            real_acc = ( label_real > 0.5 ).float().sum() / real_imgs.shape[ 0 ]
            gen_acc = ( label_gen < 0.5 ).float().sum() / gen_imgs.shape[ 0 ]
            d_acc = ( real_acc + gen_acc ) / 2
            d_loss.backward()
            optimizer_D.step()
                
            print ('Gen Imgs : ', gen_imgs.shape)
            #pdb.set_trace()

            print( "[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %.2f%%] [G loss: %f]" % \
                     ( epoch,
                      opt.n_epochs,
                      i,
                      len(train_loader_MNIST),
                      d_loss.item(),
                      d_acc * 100,
                      g_loss.item() ) )
            
            batches_done = epoch * len( train_loader_MNIST ) + i
            
            print ('Batches Done : ', batches_done)
            
            if batches_done % opt.sample_interval == 0:
                
                #new_gen_imgs = gen_imgs.view(1,1,32,32,32)

                new_gen_imgs_voxel = gen_imgs[0,:,:,:,:]

                new_gen_imgs_voxel = new_gen_imgs_voxel.squeeze(0)

                new_gen_imgs_voxel = new_gen_imgs_voxel.cpu().detach().numpy()

                new_gen_imgs_voxel = new_gen_imgs_voxel>0.2
                    
                print ('SHAAAPPPEEE : ', new_gen_imgs_voxel.shape)

                #pdb.set_trace()

                fig = plt.figure()
            
                ax = fig.gca(projection='3d')
                
                ax.set_aspect('equal')

                ax.voxels(new_gen_imgs_voxel, edgecolor="k")
                
                ax.view_init(elev = 60)

                plt.savefig(str(epoch) + "" + str(batches_done)+"digit.png")

                plt.show()
                
                #save_image( gen_imgs.data[ : 25 ],
                #            'images/%d.png' % batches_done,
                #            nrow=5,
                #            normalize=True )
                
                torch.save( generator, 'models/gen_%d.pt' % batches_done )
                torch.save( discriminator, 'models/dis_%d.pt' % batches_done )
if __name__ == '__main__':
    main()
