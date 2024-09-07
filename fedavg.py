import numpy as np
import os
import tomosipo as ts
from latent_fedavg import Latent_Fed_Avg as Latent
from network import SIREN
from tomosipo.torch_support import to_autograd
from tifffile import imread

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

node_num = 10
num_proj = 60
imgs = np.zeros((node_num, 1, 512, 512), dtype=np.float32)

d = imread('imgs.tif')

imgs[:,0,:,:] = d

from tomopy.misc.corr import circ_mask
imgs[:,0,:,:] = circ_mask(imgs[:,0,:,:], axis=0, ratio=1)

angles = np.linspace(0, np.pi, num_proj, endpoint=False)
vg = ts.volume(shape=(1, 512, 512), size=(1, 1, 1))
pg = ts.parallel(angles=angles, shape=(1, 512), size=(1, 1))
A = ts.operator(vg, pg)
f = to_autograd(A)

projs = np.zeros((imgs.shape[0], 1, num_proj, 512), dtype=np.float32)
fs = f #[]
for i in range(imgs.shape[0]):
    projs[i] = A(imgs[i])

data = {'projs': projs, 'recon': imgs}
model = SIREN(network_depth=8, network_width=256, network_input_size=512, network_output_size=1, device='cuda')
latent = Latent(data=data, model=model, forward_op=fs, glob_iters=100, local_epochs=300, adapt_epochs=20000,
                 node_lr=1e-05, work_dir='experiment_fedavg/', device='cuda')
latent.train()