import os
import numpy as np
import copy
from pathlib import Path
import torch
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from imageio import imwrite
from tqdm import tqdm

class Node:
    def __init__(self, id, train_data, model, forward_op, circ_mask, glob_iters, local_epochs, burnin_epochs ,node_lr, dir, device):
        self.train_data = train_data
        self.id = id
        self.latent_model = copy.deepcopy(model).to(device) # copy of the global model, updated during training
        self.forward_op = forward_op
        self.circ_mask = circ_mask
        self.local_epochs = local_epochs
        self.burnin_epochs = burnin_epochs # burin-in epochs for the KL term
        self.cur_epoch = 0
        self.psnrs = np.zeros(glob_iters)
        self.ssims = np.zeros(glob_iters)
        self.device = device
        self.dir = dir # directory to save the results
        if not Path(self.dir).exists():
            Path(self.dir).mkdir()
            imwrite(os.path.join(self.dir, 'gt.png'), self.train_data[2].cpu().numpy()[0])

        self.node_model = copy.deepcopy(model).to(device) # instantiate the node model
        self.node_lr = node_lr # learning rate for the node model
        self.optimizer1 = torch.optim.Adam(self.node_model.parameters(), lr=self.node_lr, betas=(0.9, 0.999), weight_decay=0)

    def train(self):
        self.node_model.train()

        X, y, gt = self.train_data # X: projection data, y: node output, gt: ground truth

        pbar = tqdm(range(self.local_epochs), leave=False)
        for r in pbar:
            # sample the epsilons
            epsilons = self.node_model.sample_epsilons(self.node_model.layer_param_shapes)
            # reparametrization trick, refer to the manuscript for details
            layer_params1 = self.node_model.transform_gaussian_samples(
                self.node_model.mus, self.node_model.rhos, epsilons)
            # forward pass with sampled parameters
            node_output = self.node_model.net(X, layer_params1)
            del layer_params1, epsilons
            node_output = node_output.squeeze(-1) * self.circ_mask

            # forward project the prediction
            node_output = self.forward_op(node_output)
            tar = y
            # compute the loss, refer to the manuscript for details
            node_loss = self.node_model.combined_loss_node(
                    output=node_output, label=tar,
                    mus=self.latent_model.mus,sigmas=[F.softplus(t) for t in self.latent_model.rhos],
                    mus_local=self.node_model.mus, sigmas_local=[F.softplus(t) for t in self.node_model.rhos],
                    factor= 0 if self.cur_epoch <= self.burnin_epochs[0] else (self.cur_epoch - self.burnin_epochs[0]) / self.burnin_epochs[1] if self.burnin_epochs[0] < self.cur_epoch < self.burnin_epochs[1] else 1)

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

        output = self.node_model.net(X, self.node_model.mus)
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
            f.write(" ".join(str(item) for item in self.psnrs))
            f.write("\nSSIM: \n")
            f.write(" ".join(str(item) for item in self.ssims))

    def save(self):
        torch.save(self.node_model.state_dict(), os.path.join(self.dir, 'model'+str(self.id)+'.pt'))
    
    def load(self, path):
        self.node_model.load_state_dict(torch.load(os.path.join(self.dir, path)))