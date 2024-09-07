import numpy as np
import os
import tomosipo as ts
from latent import Latent
from network import bSiren
from tomosipo.torch_support import to_autograd
# from imageio import imread
from tifffile import imread
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

img_size = (512, 512)

node_num = 10 # number of joint reconstruction nodes
num_proj = 60 # number of projections
imgs = np.zeros((node_num, 1, img_size[0], img_size[1]), dtype=np.float32)

d = imread('imgs.tif')

imgs[:,0,:,:] = d

# crop the center part of the image
from tomopy.misc.corr import circ_mask
imgs[:,0,:,:] = circ_mask(imgs[:,0,:,:], axis=0, ratio=1)

# cerate the forward operator
angles = np.linspace(0, np.pi, num_proj, endpoint=False)
vg = ts.volume(shape=(1, img_size[0], img_size[1]), size=(1, 1, 1))
pg = ts.parallel(angles=angles, shape=(1, img_size[0]), size=(1, 1))
A = ts.operator(vg, pg)
f = to_autograd(A)

# generate the projection data
projs = np.zeros((imgs.shape[0], 1, num_proj, img_size[0]), dtype=np.float32)
for i in range(imgs.shape[0]):
    proj = A(imgs[i])
    projs[i] = proj

# warp up the data
data = {'projs': projs, 'recon': imgs}
# cerate the baysian model
model = bSiren(network_depth=8, network_width=256, network_input_size=512, network_output_size=1, 
               weight_scale=0, rho_offset=-10, lambda_kl=1e-16, device='cuda')
# cerate the latent variable    
latent = Latent(data=data, model=model, forward_op=f, glob_iters=100, local_epochs=300, burnin_epochs= (0,10000),
                 node_lr=1e-5, work_dir='experiment_our/', device='cuda')
latent.train()