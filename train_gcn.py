import numpy as np
import torch

from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from chem_gcn.model import ChemGCN
from chem_gcn.utils import train_model, test_model, parity_plot, loss_curve, Standardizer
from chem_gcn.graphs import GraphData, collate_graph_dataset


np.random.seed(42)
torch.manual_seed(42)
use_GPU = torch.cuda.is_available()

max_atoms = 200
node_vec_len = 60
train_size = 0.7
batch_size = 32
hidden_nodes = 60
n_conv_layers = 4
n_hidden_layers = 2
learning_rate = 0.01
n_epochs = 50

main_path = Path(__file__).resolve().parent
data_path = main_path / 'data' / 'solubility_data.csv'
dataset = GraphData(
    dataset_path=data_path, max_atoms=max_atoms,
    node_vec_len=node_vec_len
)

dataset_indices = np.arange(len(dataset))
train_size = int(np.round(train_size * len(dataset)))
test_size = len(dataset) - train_size

train_indices = np.random.choice(dataset_indices, size=train_size, replace=False)
test_indices = np.array(list(set(dataset_indices) - set(train_indices)))

train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)
train_loader = DataLoader(dataset, batch_size=batch_size,
                          sampler=train_sampler, 
                          collate_fn=collate_graph_dataset)
test_loader = DataLoader(dataset, batch_size=batch_size,
                         sampler=test_sampler,
                         collate_fn=collate_graph_dataset)

model = ChemGCN(node_vec_len=node_vec_len,
                node_fea_len=hidden_nodes,
                hidden_fea_len=hidden_nodes,
                n_conv=n_conv_layers,
                n_hidden=n_hidden_layers,
                n_outputs=1,
                p_dropout=0.1)

if use_GPU:
    model.cuda()

outputs = [dataset[i][1] for i in range(len(dataset))]
standardizer = Standardizer(torch.Tensor(outputs))

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.MSELoss()

loss, mae, epoch = [], [], []

for i in range(n_epochs):
    epoch_loss, epoch_mae = train_model(i, model,
                                        train_loader,
                                        optimizer,
                                        criterion,
                                        standardizer,
                                        use_GPU,
                                        max_atoms,
                                        node_vec_len)
    loss.append(epoch_loss)
    mae.append(epoch_mae)
    epoch.append(i)

test_loss, test_mae = test_model(model, test_loader,
                                 criterion, 
                                 standardizer,
                                 use_GPU, max_atoms,
                                 node_vec_len)

print(f'Training Loss: {loss[-1]:.2f}')
print(f'Training MAE: {mae[-1]:.2f}')
print(f'Test Loss: {test_loss:.2f}')
print(f'Test MAE: {test_mae:.2f}')

plot_dir = main_path / 'plots'
parity_plot(plot_dir, model, test_loader,
            standardizer, use_GPU, 
            max_atoms, node_vec_len)

loss_curve(plot_dir, epoch, loss)