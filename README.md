# 3D_GAN

• Implementation of 3D GAN for generating voxelized MNIST digits

• 2D MNIST digits were converted to 3D voxels by stacking image pixels across the third dimension

• Also, involved implementing linear interpolation in latent space using the follwoing : 

   a) Find two randomly sampled latent vectors (say z1, z2).
  
   b) Load a pretrained model and then compute the difference between the z1 and z2 (diff). Define a step size and compute as : new z = (z1 + (diff/ step size)). 
  
   c) Perform previous step recursively and pass the new z value to the pretrained model.
  
 ## Results:
 
 ### Following are the results from training the GAN with lr = 0.0001
<p align="center">
  <img src="/images/img1.JPG">
</p>
 
<p align="center">
  <img src="/images/img2.JPG">
</p>

 ### Following are the results from training the GAN with lr = 0.004
<p align="center">
  <img src="/images/img3.JPG">
</p>

## Interpolation Results (between digits 7 and 0): 

<p align="center">
  <img src="/images/img4.JPG">
</p>
