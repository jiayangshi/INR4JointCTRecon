import torch 
import numpy as np
from tomopy.misc.corr import circ_mask

# create circular mask
def compute_mask(img):
    mask = np.ones(img.shape)
    mask = circ_mask(mask, axis=0, ratio=1)
    mask = torch.tensor(mask, dtype=torch.float32)
    return mask

# create 3D grid
def create_grid_3d(c, h, w):
    grid_z, grid_y, grid_x = torch.meshgrid([torch.linspace(0, 1, steps=c), \
                                            torch.linspace(0, 1, steps=h), \
                                            torch.linspace(0, 1, steps=w)])
    grid = torch.stack([grid_z, grid_y, grid_x], dim=-1)
    return grid

class PositionalEncoder():
    def __init__(self, embedding_size=256, coordinates_size=2, scale=4, device='cuda'):
        self.B = torch.randn((embedding_size, coordinates_size)) * scale
        self.B = self.B.to(device)

    def embedding(self, x):
        x_embedding = (2. * np.pi * x) @ self.B.t()
        x_embedding = torch.cat([torch.sin(x_embedding), torch.cos(x_embedding)], dim=-1)
        return x_embedding
    
def inverse_softplus(x):
    # softplus y = log(1+exp(x))
    # inverse softplus x = log(exp(y)-1)
    return torch.log(torch.exp(x) - 1)
