import os
import numpy as np
import copy
from pathlib import Path
import torch
from torch.autograd import Variable
from torch.nn import Module
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from imageio import imwrite
from tqdm import tqdm
import matplotlib.pyplot as plt

class Node_Fed_Avg:
    def __init__(self, id, train_data, model, forward_op, circ_mask, glob_iters, local_epochs, node_lr, dir, device):
        self.train_data = train_data
        self.id = id
        self.latent_model = copy.deepcopy(model).to(device)
        self.forward_op = forward_op
        self.circ_mask = circ_mask
        self.local_epochs = local_epochs
        self.cur_epoch = 0
        self.psnrs = np.zeros(glob_iters)
        self.ssims = np.zeros(glob_iters)
        self.device = device
        self.dir = dir
        if not Path(self.dir).exists():
            Path(self.dir).mkdir()
            imwrite(os.path.join(self.dir, 'gt.png'), self.train_data[2].cpu().numpy()[0])

        self.node_model = copy.deepcopy(model).to(device)
        
        self.node_lr = node_lr
        self.optimizer1 = torch.optim.Adam(self.node_model.parameters(), lr=self.node_lr, betas=(0.9, 0.999), weight_decay=0)

    def train(self):
        # for fed_avg, assign the weights of the global model to the local model
        latent_weights = self.latent_model.state_dict()
        self.node_model.load_state_dict(latent_weights)
        self.node_model.train()

        X, y, gt = self.train_data

        pbar = tqdm(range(self.local_epochs), leave=False)
        for r in pbar:
            ### node model
            node_output = self.node_model(X)
            node_output = node_output.squeeze(-1) * self.circ_mask
            node_output = self.forward_op(node_output)
            tar = y

            node_loss = self.node_model.direct_loss_node(
                    node_output, tar) 
            self.optimizer1.zero_grad()
            node_loss.backward()
            self.optimizer1.step()
            pbar.set_description(f"node number {self.id}: loss: {node_loss.item():.4f}")

        self.cur_epoch += self.local_epochs

    def adapt(self):
        # for final adaptation, single inr reconstruction
        self.node_model.train()

        X, y, gt = self.train_data

        pbar = tqdm(range(self.local_epochs), leave=False)
        for r in pbar:
            ### node model
            node_output = self.node_model(X)
            node_output = node_output.squeeze(-1) * self.circ_mask
            node_output = self.forward_op(node_output)
            tar = y

            node_loss = self.node_model.direct_loss_node(
                    node_output, tar) 
            self.optimizer1.zero_grad()
            node_loss.backward()
            self.optimizer1.step()
            pbar.set_description(f"node number {self.id}: loss: {node_loss.item():.4f}")

        self.cur_epoch += self.local_epochs

    @torch.no_grad()
    def set_parameters(self, model):
        # update the global model (self.model)       
        for old_param, new_param in zip(self.latent_model.parameters(), model.parameters()):
            old_param.data = new_param.data.clone()

    def get_parameters(self):
        for param in self.node_model.parameters():
            param.detach()
        return self.node_model.parameters()
    
    def get_named_parameters(self):
        for param in self.node_model.parameters():
            param.detach()
        return self.node_model.named_parameters()
    
    @torch.no_grad()
    def evaluate(self):
        self.node_model.eval()
        X, y, gt = self.train_data
        output = self.node_model(X)
        output = output.squeeze(-1) * self.circ_mask
        output = output.detach().cpu().numpy()
        torch.cuda.empty_cache()
        gt = gt.detach().cpu().numpy()
        vrange = np.max(gt) - np.min(gt)
        psnr_val = psnr(gt, output, data_range=vrange)
        ssim_val = ssim(gt.squeeze(0), output.squeeze(0), data_range=vrange)
        print(f"node number {self.id}: PSNR: {psnr_val:.2f} SSIM: {ssim_val:.2f}")
        self.psnrs[self.cur_epoch//self.local_epochs-1] = psnr_val
        self.ssims[self.cur_epoch//self.local_epochs-1] = ssim_val
        imwrite(os.path.join(self.dir, str(self.cur_epoch)+'.png'), output.squeeze(0))

        # write metrics file
        with open(os.path.join(self.dir,'metric.txt'), 'w') as f:
            f.write("PSNR: \n")
            f.write(" ".join(f'{item:.2f}' for item in self.psnrs))
            f.write("\nSSIM: \n")
            f.write(" ".join(f'{item:.3f}' for item in self.ssims))


    def save(self):
        torch.save(self.node_model.state_dict(), os.path.join(self.dir, 'model'+str(self.id)+'.pt'))
    
    def load(self, path):
        self.node_model.load_state_dict(torch.load(os.path.join(self.dir, path)))