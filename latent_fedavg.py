import os
import torch
import copy
from pathlib import Path
from tqdm import tqdm
from node_fedavg import Node_Fed_Avg as Node
from imageio import imwrite
from utils import create_grid_3d, PositionalEncoder, compute_mask

class Latent_Fed_Avg:
    def __init__(self, data, model, forward_op, glob_iters, local_epochs, adapt_epochs=10000,node_lr=0.00001, work_dir='exp', device='cuda',pos_encoder=None,latent_model=None):
        self.data = data
        self.model = copy.deepcopy(model).to(device)
        if latent_model is not None:
            print("Loading latent model...")
            self.load_model(latent_model)
        self.forward_op = forward_op
        self.cur_iter = 0
        self.glob_iters = glob_iters
        self.local_epochs = local_epochs
        assert adapt_epochs <= glob_iters * local_epochs
        self.adapt_epochs = adapt_epochs
        self.node_lr = node_lr
        self.work_dir = work_dir
        if not Path(self.work_dir).exists():
            print("Creating folder for working directory")
            Path(self.work_dir).mkdir()
        self.pos_encoder = PositionalEncoder(model.network_width, 3, 4, device)
        if pos_encoder is not None:
            print("Loading positional encoder...")
            self.load_pos_encoder(pos_encoder)
        self.save_pos_encode()
        self.circ_mask = compute_mask(self.data['recon'][0]).to(device)
        self.device = device
        self.num_node = data['projs'].shape[0]
        tar_size = (data['recon'].shape[1], data['recon'].shape[2], data['recon'].shape[3])
        self.encoded_pos = self.pos_encoder.embedding(create_grid_3d(tar_size[0], tar_size[1], tar_size[2]).to(device))
        self.nodes = []
        print('nodes initializting...')
        
        for i in tqdm(range(self.num_node)):
            train_data = (self.encoded_pos.to(self.device), torch.tensor(data['projs'][i], dtype=torch.float32).to(self.device),torch.tensor(data['recon'][i]).to(self.device))
            
            # check is forward_op is a list
            if type(self.forward_op) == list:
                cur_forward_op = self.forward_op[i]
            else:
                cur_forward_op = self.forward_op
            
            node = Node(i, train_data, model, cur_forward_op, self.circ_mask, glob_iters, local_epochs, node_lr, os.path.join(work_dir, str(i)), device)

            self.nodes.append(node)
        print(f'nodes initialized, total {self.num_node} nodes')
        with open(os.path.join(self.work_dir,'config.txt'), 'w') as file:
            file.write("glob_iters:" + str(glob_iters) + '\n')
            file.write("local_epochs:" + str(local_epochs) + '\n')
            file.write("adapt_epochs:" + str(adapt_epochs) + '\n')
            file.write("node_lr:" + str(node_lr) + '\n')

    def train(self):
        for iter in range(self.glob_iters):
            if iter*self.local_epochs < self.glob_iters*self.local_epochs - self.adapt_epochs:
                self.cur_iter = iter
                print(f"global iter {iter+1}...")
                self.send_parameters()

                for node in self.nodes:
                    node.train()
                    node.evaluate()
                    node.save()
                    torch.cuda.empty_cache()

                self.aggregate_parameters()
                self.evaluate()
                self.save_latent_model()
                torch.cuda.empty_cache()
            else:
                self.cur_iter = iter
                print(f"global iter {iter+1}...")
                for node in self.nodes:
                    node.adapt()
                    node.evaluate()
                    node.save()
                    torch.cuda.empty_cache()


    def send_parameters(self):
        # send latent parameters to nodes before training on single node
        assert (self.nodes is not None and len(self.nodes) > 0)
        for node in self.nodes:
            node.set_parameters(self.model)
   
    @torch.no_grad()
    def aggregate_parameters(self):
        assert (self.num_node is not None and self.num_node > 0)
        for name, param in self.model.named_parameters():
            param.data = torch.zeros_like(param.data)

        for node in self.nodes:
            for (name, param), (n_name, n_param) in zip(self.model.named_parameters(), node.node_model.named_parameters()):
                param.add_(n_param / self.num_node)


    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        output = self.model(self.encoded_pos)
        output = output.squeeze(-1) * self.circ_mask
        imwrite(os.path.join(self.work_dir, 'latent_'+str(self.cur_iter)+'.png'), output.detach().cpu().numpy()[0, :, :])
        torch.cuda.empty_cache()

    def save_pos_encode(self):
        torch.save(self.pos_encoder.B, os.path.join(self.work_dir, 'pos_encoder.pth'))
    
    def save_latent_model(self):
        torch.save(self.model.state_dict(), os.path.join(self.work_dir, 'latent_model.pth'))

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
    
    def load_pos_encoder(self, path):
        self.pos_encoder.B = torch.load(path)