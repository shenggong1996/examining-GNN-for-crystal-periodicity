# model
from ast import Global
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as tg
import torch_scatter
import e3nn
from e3nn import o3
from typing import Dict, Union

# crystal structure data
from ase import Atom, Atoms
from ase.neighborlist import neighbor_list
from ase.visualize.plot import plot_atoms

# data pre-processing and visualization
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn import metrics
import pandas as pd

# utilities
import time
import json
import os
from tqdm import tqdm
from utils.utils_data import (deload_data, train_valid_test_split, plot_example, plot_predictions, plot_partials,
                              palette, colors, cmap)
from utils.utils_model import Network, visualize_layers, train
from utils.utils_plot import plotly_surface, plot_orbitals, get_middle_feats

bar_format = '{l_bar}{bar:10}{r_bar}{bar:-10b}'
default_dtype = torch.float64
torch.set_default_dtype(default_dtype)
print (torch.cuda.is_available())

# load data
df, species, mean, std = deload_data('/root/cv_training.csv') # address to training set
df_test, species_test, _, _ = deload_data('/root/cv_test.csv') # address to test set
run_name = 'model_de_cv_' + time.strftime("%y%m%d", time.localtime())
print(run_name)

# encoding atom type and mass
type_encoding = {}
specie_am = []
for Z in tqdm(range(1, 119), bar_format=bar_format):
    specie = Atom(Z)
    type_encoding[specie.symbol] = Z - 1
    specie_am.append(specie.mass)

type_onehot = torch.eye(len(type_encoding))
am_onehot = torch.diag(torch.tensor(specie_am))

# build data
def build_data(entry, type_encoding, type_onehot, r_max=5.):
    symbols = list(entry.structure.symbols).copy()
    idxs = torch.LongTensor([entry.idx for _ in range(len(symbols))])
    positions = torch.from_numpy(entry.structure.positions.copy())
    lattice = torch.from_numpy(entry.structure.cell.array.copy()).unsqueeze(0)
    feature = torch.tensor(entry.feature)

    # edge_src and edge_dst are the indices of the central and neighboring atom, respectively
    # edge_shift indicates whether the neighbors are in different images or copies of the unit cell
    edge_src, edge_dst, edge_shift = neighbor_list("ijS", a=entry.structure, cutoff=r_max, self_interaction=True)
    
    # compute the relative distances and unit cell shifts from periodic boundaries
    edge_batch = positions.new_zeros(positions.shape[0], dtype=torch.long)[torch.from_numpy(edge_src)]
    edge_vec = (positions[torch.from_numpy(edge_dst)]
                - positions[torch.from_numpy(edge_src)]
                + torch.einsum('ni,nij->nj', torch.tensor(edge_shift, dtype=default_dtype), lattice[edge_batch]))

    # compute edge lengths (rounded only for plotting purposes)
    edge_len = np.around(edge_vec.norm(dim=1).numpy(), decimals=2)
    
    data = tg.data.Data(
        pos=positions, lattice=lattice, symbol=symbols,
        x=am_onehot[[type_encoding[specie] for specie in symbols]],   # atomic mass (node feature)
        z=type_onehot[[type_encoding[specie] for specie in symbols]], # atom type (node attribute),
        edge_index=torch.stack([torch.LongTensor(edge_src), torch.LongTensor(edge_dst)], dim=0),
        edge_shift=torch.tensor(edge_shift, dtype=default_dtype),
        edge_vec=edge_vec, edge_len=edge_len,
        phdos= (torch.tensor(entry.prop) - mean) / std,
        idxs = idxs,
        feature = feature
    )
    
    return data

r_max = 8. # cutoff radius
df['data'] = df.progress_apply(lambda x: build_data(x, type_encoding, type_onehot, r_max), axis=1)
df_test['data'] = df_test.progress_apply(lambda x: build_data(x, type_encoding, type_onehot, r_max), axis=1)

idx_train, idx_valid, idx_test = train_valid_test_split(df, species, valid_size=.2, test_size=.01, seed=42, plot=False)

# format dataloaders
batch_size = 16
dataloader_train = tg.loader.DataLoader(df.iloc[idx_train]['data'].values, batch_size=batch_size, shuffle=True)
dataloader_valid = tg.loader.DataLoader(df.iloc[idx_valid]['data'].values, batch_size=batch_size)
#dataloader_test = tg.loader.DataLoader(df.iloc[idx_test]['data'].values, batch_size=batch_size)
dataloader_test = tg.loader.DataLoader(df_test['data'].values, batch_size=batch_size)

class PeriodicNetwork(Network):
    def __init__(self, in_dim, em_dim, device, **kwargs):            
        # override the `reduce_output` keyword to instead perform an averge over atom contributions    
        self.pool = False
        self.device = device
        if kwargs['reduce_output'] == True:
            kwargs['reduce_output'] = False
            self.pool = True
            
        super().__init__(**kwargs)

        # embed the mass-weighted one-hot encoding
        self.em = nn.Linear(in_dim, em_dim)
        self.batch_norm = nn.BatchNorm1d(28)
        self.fea_em = nn.Sequential(nn.Linear(28,28),
                                         nn.SiLU())
        self.fea_update = nn.Sequential(nn.Linear(out_dim+28, out_dim+28),
                                         nn.SiLU(),
                                         nn.Linear(out_dim+28, out_dim+28),
                                         nn.SiLU(),
                                         nn.Linear(out_dim+28, 1))

    def forward(self, data: Union[tg.data.Data, Dict[str, torch.Tensor]]) -> torch.Tensor:
        data.x = F.relu(self.em(data.x))
        data.z = F.relu(self.em(data.z))
        output = super().forward(data)
        
        # if pool_nodes was set to True, use scatter_mean to aggregate
        if self.pool == True:
            index = data.idxs.view(-1)
            unique_vals = torch.unique_consecutive(index)
            mapping = {val.item(): i for i, val in enumerate(unique_vals)}
            transformed_index = torch.tensor([mapping[val.item()] for val in index], device=self.device)
            output = torch_scatter.scatter_mean(output, transformed_index, dim=0)  # take mean over atoms per example

        first_dim_size = output.shape[0]
        fea = data.feature.reshape(first_dim_size, 28)
        fea = self.fea_em(self.batch_norm(fea))        
        output = torch.cat((output, fea), dim=1)
        output = self.fea_update(output).view(-1)
        
        return output

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print('torch device:' , device)

out_dim = 64
em_dim = 64  

model = PeriodicNetwork(
    in_dim=118,                            # dimension of one-hot encoding of atom type
    em_dim=em_dim,                         # dimension of atom-type embedding
    device=device,                         
    irreps_in=str(em_dim)+"x0e",           # em_dim scalars (L=0 and even parity) on each atom to represent atom type
    irreps_out=str(out_dim)+"x0e",         # out_dim scalars (L=0 and even parity) to output
    irreps_node_attr=str(em_dim)+"x0e",    # em_dim scalars (L=0 and even parity) on each atom to represent atom type
    layers=2,                              # number of nonlinearities (number of convolutions = layers + 1)
    mul=32,                                # multiplicity of irreducible representations
    lmax=2,                                # maximum order of spherical harmonics
    max_radius=r_max,                      # cutoff radius for convolution
    num_neighbors=20,          # scaling factor based on the typical number of neighbors
    reduce_output=True                     # whether or not to aggregate features of all atoms at the end
)

print(model)

opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.95)

loss_fn = torch.nn.MSELoss()
loss_fn_mae = torch.nn.L1Loss()

model.pool = True
train(model, opt, dataloader_train, dataloader_valid, loss_fn, loss_fn_mae, run_name,
      max_iter=300, scheduler=scheduler, device=device)

model.eval()

trues = []
preds = []
ids = []
with torch.no_grad():
    for i, d in tqdm(enumerate(dataloader_test), total=len(dataloader_test), bar_format=bar_format):
        d.to(device)
        output = model(d)
        trues += d.phdos.cpu().tolist()
        preds += output.cpu().tolist()

trues = np.array(trues)*std + mean
preds = np.array(preds)*std + mean

print('MAE:', metrics.mean_absolute_error(trues, preds))  
print('MAE/MAD', metrics.mean_absolute_error(trues, preds)/np.mean(np.abs(trues - np.mean(trues))))
print('R2:', metrics.r2_score(trues, preds))

save = {'id':df_test['ids'].values,
        'true':trues.tolist(),
        'pred':preds.tolist()}

save = pd.DataFrame(save)
save.to_csv(run_name + '.csv')


