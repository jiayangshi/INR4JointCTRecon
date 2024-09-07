import os
import torch
from torch import nn
import numpy as np
import tomosipo as ts
from pathlib import Path
from tomosipo.torch_support import to_autograd
from network import SIREN, SirenLayer
from utils import create_grid_3d, PositionalEncoder, compute_mask, inverse_softplus
from imageio import imwrite
from tqdm import tqdm
from tifffile import imread
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

class INRw():
    def __init__(self, data, forward_op, dir='wild',network_depth=8, network_width=256, network_input_size=512, network_output_size=1, t_embedding_dim=16,
                  iters=30000, eval_per_iters=1000, static_iters=30000,device='cuda'):
        self.data = data
        self.dir = dir
        self.eval_per_iters = eval_per_iters
        self.static_iters = static_iters
        if not Path(self.dir).exists():
            Path(self.dir).mkdir()
        self.static_model = SIREN(network_depth, network_width, network_input_size, network_output_size+network_input_size//2)
        self.static_model = self.static_model.to(device)
        self.pos_encoder = PositionalEncoder(network_width, 3, 4, device)
        self.loss =  torch.nn.MSELoss()
        self.num_node = data['projs'].shape[0]
        self.forward_op = forward_op
        self.circ_mask = compute_mask(self.data['recon'][0]).to(device)
        self.device = device
        self.tar_size = (data['recon'].shape[1], data['recon'].shape[2], data['recon'].shape[3])
        self.encoded_pos = self.pos_encoder.embedding(create_grid_3d(self.tar_size[0], self.tar_size[1], self.tar_size[2]).to(device))
        self.indicies = torch.zeros((self.num_node, data['recon'].shape[1], data['recon'].shape[2], data['recon'].shape[3])).long().to(device)
        self.transient_embedding = nn.Embedding(num_embeddings=100, embedding_dim=t_embedding_dim).to(device)
        self.transient_models = []
        for i in range(self.num_node):
            node_dir = os.path.join(self.dir,str(i))
            if not Path(node_dir).exists():
                Path(node_dir).mkdir()
                imwrite(os.path.join(node_dir, 'gt.png'), self.data['recon'][i,0])
            self.transient_models.append(nn.Sequential(
                                        SirenLayer(t_embedding_dim+network_input_size//2, network_input_size//2),
                                        SirenLayer(network_input_size//2, network_input_size//2),
                                        SirenLayer(network_input_size//2, network_input_size//2),
                                        SirenLayer(network_input_size//2, network_output_size, is_last=True), nn.Sigmoid()).to(device))    
        self.parm_to_optim = []
        self.parm_to_optim.extend(self.static_model.parameters())
        self.parm_to_optim.extend(self.transient_embedding.parameters())
        for i in range(self.num_node):
            self.parm_to_optim.extend(self.transient_models[i].parameters())
        self.optimizer = torch.optim.Adam(self.parm_to_optim, lr=1e-5 , betas=(0.9, 0.999), weight_decay=0)
        self.iters = iters
        self.psnrs = np.zeros((self.num_node, self.iters//self.eval_per_iters))
        self.ssims = np.zeros((self.num_node, self.iters//self.eval_per_iters))
    

    def train(self):
        pbar = tqdm(range(self.iters), leave=False)
        for iter in pbar:
            static_output = self.static_model(self.encoded_pos)
            static_density, inter = static_output[:,:,:,0], static_output[:,:,:,1:]

            static_embedding = self.transient_embedding(self.indicies)

            loss = 0.0

            for n in range(self.num_node):
                transient_input = torch.cat([static_embedding[n], inter], dim=-1)
                transient_density = self.transient_models[n](transient_input).squeeze(-1)
                transient_density=static_density+transient_density

                train_projs = self.forward_op(transient_density)
                loss = loss+self.loss(train_projs, self.data['projs'][n])
            
            self.optimizer.zero_grad()
            loss.backward()
            if iter < self.static_iters:
                self.optimizer.step()
            else:
                for param in self.static_model.parameters():
                    param.requires_grad = False
                self.optimizer.step()
            pbar.set_description(f"Loss:{loss.item():.4f}")

            if (iter+1) % self.eval_per_iters == 0:
                self.test(iter)
                self.save()


    @torch.no_grad()
    def test(self, idx):
        static_output = self.static_model(self.encoded_pos)
        static_density, inter = static_output[:,:,:,0], static_output[:,:,:,1:]

        static_embedding = self.transient_embedding(self.indicies)
        imwrite(os.path.join(self.dir,f'static_density_{idx+1}.png'), static_density[0].cpu().numpy())

        for n in range(self.num_node):
            transient_input = torch.cat([static_embedding[n], inter], dim=-1)
            transient_density = self.transient_models[n](transient_input).squeeze(-1)
            transient_density=static_density+transient_density

            imwrite(os.path.join(self.dir,str(n),f'density_{idx+1}.png'), transient_density[0].cpu().numpy())
            imwrite(os.path.join(self.dir,str(n),f'transient_{idx+1}.png'), (transient_density-static_density)[0].cpu().numpy())
            vmin, vmax = self.data['recon'][n].min(), self.data['recon'][n].max()
            p = psnr(self.data['recon'][n,0], transient_density[0].cpu().numpy(), data_range=vmax-vmin)
            s = ssim(self.data['recon'][n,0], transient_density[0].cpu().numpy(), data_range=vmax-vmin)
            self.psnrs[n, (idx+1)//self.eval_per_iters] = p
            self.ssims[n, (idx+1)//self.eval_per_iters] = s
            with open(os.path.join(self.dir, str(n), 'metric.txt'), 'w') as f:
                f.write("PSNR: \n")
                f.write(" ".join(str(item) for item in self.psnrs[n]))
                f.write("\nSSIM: \n")
                f.write(" ".join(str(item) for item in self.ssims[n]))
            
    def save(self):
        torch.save(self.static_model.state_dict(), os.path.join(self.dir,'static_model.pth'))
        torch.save(self.pos_encoder.B, os.path.join(self.dir, 'pos_encoder.pth'))
        for n in range(self.num_node):
            torch.save(self.transient_models[n], os.path.join(self.dir,str(n),f't_model{n}.pth'))

node_num=10
num_proj=60

imgs = np.zeros((node_num,1,512,512))
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


data = {'projs': torch.tensor(projs, dtype=torch.float32).to('cuda'), 'recon': imgs}

inrw = INRw(data=data, forward_op=f, iters=60000, dir='experiment_wild', eval_per_iters=1000, static_iters=20000, 
            network_depth=8, network_width=256, network_input_size=512, network_output_size=1, t_embedding_dim=16, device='cuda')
inrw.train()