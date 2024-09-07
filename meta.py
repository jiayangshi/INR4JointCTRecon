import os
import torch
import tomosipo as ts
import copy
import sys
from pathlib import Path
import numpy as np
from network import SIREN
from tqdm import tqdm
from tifffile import imread
from imageio import imwrite
from tomosipo.torch_support import to_autograd
from utils import compute_mask, create_grid_3d, PositionalEncoder
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# reference: https://github.com/tancik/learnit?tab=readme-ov-file
def inner_loop(model, optim, tar, coords, inner_steps=2):
    """
    train the inner model for a specified number of iterations
    """
    for step in range(inner_steps):        
        optim.zero_grad()
        pred = model(coords)
        proj_pred = f(circ_mask*pred.squeeze(-1))
        inner_loss = loss(proj_pred, tar)

        inner_loss.backward()
        optim.step()

def train_meta(meta_model, meta_optim, data, strategy, device):
    """
    train the meta_model for one epoch using MAML or REPTILE meta learning

    """
    assert strategy in ['MAML', 'REPTILE']
    if strategy == 'MAML':
        for i in range(len(data)):
            coord, proj, img = data[i]
            coord, proj, img = coord.to(device), proj.to(device), img.to(device)

            meta_optim.zero_grad()

            inner_model = copy.deepcopy(meta_model)
            inner_optim = torch.optim.SGD(inner_model.parameters(), lr=inner_lr)
            inner_loop(inner_model, inner_optim, proj, coord, inner_steps=inner_steps)

            with torch.no_grad():
                for meta_param, inner_param in zip(meta_model.parameters(), inner_model.parameters()):
                    meta_param.grad = meta_param - inner_param
            
            meta_optim.step()
    else:
        raise NotImplementedError

        

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using {} device'.format(device))

node_num = 10
num_proj = 60

working_dir = 'experiment_meta'
if not Path(working_dir).exists():
    print("Creating folder for working directory")
    Path(working_dir).mkdir()

# load data
imgs = np.zeros((node_num,1,512,512))

if not Path(working_dir).exists():
    print("Creating folder for working directory")
    Path(working_dir).mkdir()

d = imread('imgs.tif')

imgs[:,0,:,:] = d

# crop the center part of the image
from tomopy.misc.corr import circ_mask
imgs[:,0,:,:] = circ_mask(imgs[:,0,:,:], axis=0, ratio=1)

imgs = torch.tensor(imgs, dtype=torch.float32)

# meta learning parameters
meta_strategy = 'MAML' # 'MAML' or 'REPTILE'
meta_epochs =  1000 

outer_lr = 1e-6 
inner_lr = 0.001 
inner_steps = 10 

recon_iters = 20000

with open(os.path.join(working_dir,'config.txt'), 'w') as file:
    file.write("meta_strategy: "+meta_strategy+"\n")
    file.write("meta_epochs: "+str(meta_epochs)+"\n")
    file.write("outer_lr: "+str(outer_lr)+"\n")
    file.write("inner_lr: "+str(inner_lr)+"\n")
    file.write("inner_steps: "+str(inner_steps)+"\n")
    file.write("recon_iters: "+str(recon_iters)+"\n")

loss = torch.nn.MSELoss()

# create forward operation
angles = np.linspace(0, np.pi, num_proj, endpoint=False)
vg = ts.volume(shape=(1, 512, 512), size=(1, 1, 1))
pg = ts.parallel(angles=angles, shape=(1, 512), size=(1, 1))
A = ts.operator(vg, pg)
f = to_autograd(A)

# create circular mask
circ_mask = compute_mask(np.zeros((1,512,512)))
circ_mask = circ_mask.to(device)

# positional encoder
encoder = PositionalEncoder(embedding_size=256, coordinates_size=3, scale=4)
grid = create_grid_3d(imgs.shape[1],imgs.shape[2],imgs.shape[3])
grid = grid.to(device)
embedding = encoder.embedding(grid)
torch.save(encoder.B, os.path.join(working_dir, 'pos_encoder.pth'))

model = SIREN(network_depth=8, network_width=256, network_input_size=512, network_output_size=1)
optimizer = torch.optim.Adam(model.parameters(), lr=outer_lr)
model = model.to(device)

# create the projection data
projs = []
for i in range(imgs.shape[0]):
    projs.append(f(imgs[i]))

embedding = embedding.to(device)
data = []
for i in range(imgs.shape[0]):
    data.append((embedding, projs[i], imgs[i]))

for epoch in tqdm(range(meta_epochs)):
    if epoch == 0 or (epoch + 1) % 1000 == 0:
        model.eval()
        with torch.no_grad():
            test_output = model(embedding)
            test_output = circ_mask*(test_output.squeeze(-1))
            imwrite(os.path.join(working_dir, 'intermidate_meta_'+str(epoch+1)+'.png'), test_output.detach().cpu().numpy()[0])
    train_meta(model, optimizer, data, meta_strategy, device)

torch.save(model.state_dict(), os.path.join(working_dir, 'weights_meta_prior.pt'))

for i in range(imgs.shape[0]):
    node_dir = os.path.join(working_dir, str(i))
    if not Path(node_dir).exists():
        print("Creating folder for sub recon directory")
        Path(node_dir).mkdir()
    imwrite(os.path.join(node_dir, 'gt.png'), imgs.numpy()[i,0])

    # use the trained model to reconstruct the images
    model2 = SIREN(network_depth=8, network_width=256, network_input_size=512, network_output_size=1)
    model2.load_state_dict(torch.load(os.path.join(working_dir,'weights_meta_prior.pt')))
    model2 = model2.to(device)
    projs_2 = projs[i].to(device)

    optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.00001 , betas=(0.9, 0.999), weight_decay=0)
    psnrs = np.zeros(recon_iters//1000)
    ssims = np.zeros(recon_iters//1000)
    for epoch in tqdm(range(recon_iters)):
        if epoch == 0 or (epoch + 1) % 1000 == 0:
            model2.eval()
            with torch.no_grad():
                test_output = model2(embedding)
                test_output = circ_mask*test_output.squeeze(-1)
                imwrite(os.path.join(node_dir, 'intermidate_'+str(epoch+1)+'.png'), test_output.detach().cpu().numpy()[0])

                vrange = imgs[i,0].detach().cpu().numpy().max()-imgs[i,0].detach().cpu().numpy().min()
                psnr_val = psnr(imgs[i,0].detach().cpu().numpy(), test_output.detach().cpu().numpy()[0], data_range=vrange)
                psnrs[(epoch + 1) // 1000-1] = psnr_val
                ssim_val = ssim(imgs[i,0].detach().cpu().numpy(), test_output.detach().cpu().numpy()[0], data_range=vrange)
                ssims[(epoch + 1) // 1000-1] = ssim_val

                with open(os.path.join(node_dir,'metrics.txt'), 'w') as file:
                    file.write("PSNR: \n")
                    file.write(" ".join(str(item) for item in psnrs))
                    file.write("\nSSIM: \n")
                    file.write(" ".join(str(item) for item in ssims))
            torch.save(model2.state_dict(), os.path.join(node_dir, 'model'+str(i)+'.pt'))
        
        model2.train()
        optimizer2.zero_grad()
        train_output = model2(embedding)  # [B, H, W, 3]
        train_projs = f(torch.mul(train_output.squeeze(-1),circ_mask))

        train_loss = loss(train_projs, projs_2)
    
        train_loss.backward()
        optimizer2.step()