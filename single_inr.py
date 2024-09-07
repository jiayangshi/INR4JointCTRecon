import os
import socket
import torch
from pathlib import Path
from tqdm import tqdm
import tomosipo as ts
import numpy as np
from network import SIREN
from torch.utils.data import DataLoader
from imageio.v2 import imread, imwrite
from PIL import Image
from tomosipo.torch_support import to_autograd
from utils import compute_mask, circ_mask, create_grid_3d, PositionalEncoder
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import sys

# Get current host
HOSTNAME = socket.gethostname()

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = 'cuda:0'
print('Using {} device'.format(device))

num_proj = 60
node_num = 10

working_dir = 'experiment_single'

if not Path(working_dir).exists():
    print("Creating folder for working directory")
    Path(working_dir).mkdir()

# load data
imgs = np.zeros((node_num,1,512,512))

d = imread('imgs.tif')

imgs[:,0,:,:] = d

from tomopy.misc.corr import circ_mask
imgs[:,0,:,:] = circ_mask(imgs[:,0,:,:], axis=0, ratio=1)

# positional enoder
encoder = PositionalEncoder(embedding_size=256, coordinates_size=3, scale=4)
torch.save(encoder.B, os.path.join(working_dir, 'pos_encoder.pth'))

grid = create_grid_3d(imgs.shape[1],imgs.shape[2],imgs.shape[3])
grid = grid.to(device)
embedding = encoder.embedding(grid)

epochs = 30000

for i in range(imgs.shape[0]):
    node_dir = os.path.join(working_dir, str(i))
    if not Path(node_dir).exists():
        print("Creating folder for sub recon directory")
        Path(node_dir).mkdir()
        imwrite(os.path.join(node_dir, 'gt.png'), imgs[i][0])

    model = SIREN(network_depth=8, network_width=256, network_input_size=512, network_output_size=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001 , betas=(0.9, 0.999), weight_decay=0)
    model = model.to(device)

    image = torch.tensor(imgs[i], dtype=torch.float32)
    circ_mask = compute_mask(image)

    angles = np.linspace(0, np.pi, num_proj, endpoint=False)
    vg = ts.volume(shape=(1, 512, 512), size=(1, 1, 1))
    pg = ts.parallel(angles=angles, shape=(1, 512), size=(1, 1))
    A = ts.operator(vg, pg)
    f = to_autograd(A)

    projs = f(image)
    loss = torch.nn.MSELoss()

    grid = grid.to(device)
    image = image.to(device)
    projs = projs.to(device)
    embedding = embedding.to(device)
    circ_mask = circ_mask.to(device)
    encoder.B = encoder.B.to(device)

    psnrs = np.zeros(epochs//1000)
    ssims = np.zeros(epochs//1000)
    for epoch in tqdm(range(epochs)):
        if epoch == 0 or (epoch + 1) % 1000 == 0:
            model.eval()
            with torch.no_grad():
                test_output = model(embedding)

                test_output = torch.mul(test_output.squeeze(-1), circ_mask)
                imwrite(os.path.join(node_dir, 'intermidate_'+str(epoch+1)+'.png'), test_output.detach().cpu().numpy()[0])

                vrange = image.detach().cpu().numpy()[0].max()-image.detach().cpu().numpy()[0].min()
                psnrs[(epoch + 1) // 1000-1] = psnr(image.detach().cpu().numpy()[0], test_output.detach().cpu().numpy()[0], data_range=vrange)
                ssims[(epoch + 1) // 1000-1] = ssim(image.detach().cpu().numpy()[0], test_output.detach().cpu().numpy()[0], data_range=vrange)

                print(f"epoch:{epoch+1} PSNR: {psnrs[(epoch + 1) // 1000-1]:.2f}")
                print(f"epoch:{epoch+1} SSIM: {ssims[(epoch + 1) // 1000-1]:.2f}")
                # write metrics file
                with open(os.path.join(node_dir,'single_inr_metric.txt'), 'w') as file:
                    file.write("PSNR: \n")
                    file.write(" ".join(str(item) for item in psnrs))
                    file.write("\nSSIM: \n")
                    file.write(" ".join(str(item) for item in ssims))
                
        optimizer.zero_grad()

        train_output = model(embedding)  # [B, H, W, 3]

        train_projs = f(torch.mul(train_output.squeeze(-1),circ_mask))
        train_loss = loss(train_projs, projs) 
    
        train_loss.backward()
        optimizer.step()

    
    torch.save(model.state_dict(), os.path.join(node_dir, 'single_inr_weights.pt'))