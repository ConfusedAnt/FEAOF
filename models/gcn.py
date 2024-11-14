"""
Author: Derek van Tilborg -- TU/e -- 19-05-2022

Basic graph convolutional network [1] using a transformer as global pooling [2].

1. Kipf & Welling (2017). Semi-Supervised Classification with Graph Convolutional Networks
2. Baek et al. (2021). Accurate Learning of Graph Representations with Graph Multiset Pooling

"""

from const import RANDOM_SEED
from models.utils import GNN
from torch_geometric.nn import GCNConv, GraphMultisetTransformer
from torch.nn import Linear, Dropout
import torch.nn.functional as F
import torch


class GCN(GNN):
    def __init__(self, node_feat_in: int = 37, node_hidden: int = 128, transformer_hidden: int = 128,
                 n_conv_layers: int = 4, n_fc_layers: int = 4, fc_hidden: int = 128, lr: float = 0.000005,
                 epochs: int = 1000, dropout: float = 0.2, lr_decay_ratio=0.95, *args, **kwargs):
        super().__init__()

        self.model = GCNmodel(node_feat_in=node_feat_in, node_hidden=node_hidden, n_conv_layers=n_conv_layers,
                              transformer_hidden=transformer_hidden, fc_hidden=fc_hidden, n_fc_layers=n_fc_layers,
                              dropout=dropout)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.loss_fn = torch.nn.MSELoss()
        self.loss_fn = torch.nn.BCELoss()
        # self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.epochs = epochs
        self.name = 'GCN'
        self.lr = lr
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                    # self.epochs,
                                                    step_size=3,
                                                    gamma=lr_decay_ratio,
                                                    last_epoch=-1,
                                                    verbose=False)
        # Move the whole model to the gpu
        self.model = self.model.to(self.device)

    def __repr__(self):
        return f"{self.model}"


class GCNmodel(torch.nn.Module):
    def __init__(self, node_feat_in: int = 37, n_conv_layers: int = 4, n_fc_layers: int = 1, node_hidden: int = 128,
                 fc_hidden: int = 128, transformer_hidden: int = 128, dropout: float = 0.2, seed: int = RANDOM_SEED,
                 *args, **kwargs):

        # Init parent
        super().__init__()
        torch.manual_seed(seed)

        # GCN layer(s)
        self.conv_layers = torch.nn.ModuleList()
        self.conv_layers.append(GCNConv(node_feat_in, node_hidden))
        for k in range(n_conv_layers-1):
            self.conv_layers.append(GCNConv(node_hidden, node_hidden))

        # Global pooling
        self.transformer = GraphMultisetTransformer(in_channels=node_hidden,
                                                    hidden_channels=transformer_hidden,
                                                    out_channels=fc_hidden,
                                                    num_heads=8)
            
        self.dropout = Dropout(dropout)

        self.sigmoid = torch.nn.Sigmoid()

        ## Fingerprints processing path
        self.linear1 = torch.nn.Linear(1024+881, 64)

        ## Phys_Chem
        self.phys_chem_linear = torch.nn.Linear(11, 11, bias=True)
        torch.nn.init.kaiming_normal_(self.phys_chem_linear.weight, nonlinearity="relu")
        
        ## hERG Rep Process
        self.herg_linear = torch.nn.Linear(1280, 128, bias=True)
        self.herg_out = torch.nn.Linear(fc_hidden + 128 + 128 + 64 + 11, fc_hidden)

        ## Inter Rep Process
        self.inter_linear = torch.nn.Linear(1441, 128, bias=True)
        self.inter_out = torch.nn.Linear(fc_hidden + 128 + 64 + 11, fc_hidden)

        
        self.sig_out = torch.nn.Linear(fc_hidden, 1)

        self.graph_out = torch.nn.Linear(fc_hidden, 1)

        self.fc = torch.nn.ModuleList()
        for k in range(n_fc_layers):
            self.fc.append(Linear(fc_hidden, fc_hidden))
        
    def forward(self, herg_em, x, edge_index, edge_attr, batch, mol_rep):

        # Conv layers
        h = F.relu(self.conv_layers[0](x.float(), edge_index))
        for k in range(len(self.conv_layers) - 1):
            h = F.relu(self.conv_layers[k+1](h, edge_index))

        # Global graph pooling with a transformer
        out = self.transformer(x=h, index=batch, edge_index=edge_index)


        # ## Fingerprints
        # out_fingerprints = F.relu(self.linear1(mol_rep[:,:1905]))
        # # out_fingerprints = F.relu(self.linear2(out_fingerprints)) ### 64

        # ## Phys_Chem
        # out_phys_chem = F.relu(self.phys_chem_linear(mol_rep[:,1905:1916]))  ### 11

        # cat_out = torch.cat((out, out_fingerprints), dim=1)
        # cat_out = torch.cat((cat_out, out_phys_chem), dim=1)
        # # add_out =  h + out_fingerprints   ### add

        # ## Inter Rep
        # inter_em = F.relu(self.inter_linear(mol_rep[:,3742:]))

        # out = torch.cat((cat_out, inter_em), dim=1)

        ## hERG Rep
        if herg_em is not None:
            herg_em = F.relu(self.herg_linear(herg_em))
            herg_em = herg_em.expand(out.shape[0], -1)
            out = torch.cat((out, herg_em), dim=1)
            # h = add_out + herg_em ### add
            out = F.relu(self.herg_out(out))

            for k in range(len(self.fc)):
                out = F.relu(self.fc[k](out))
                out = self.dropout(out)

            out = self.sigmoid(self.sig_out(out))
        else:

            # out = F.relu(self.inter_out(out))
            # # Apply a fully connected layer.
            # for k in range(len(self.fc)):
            #     out = F.relu(self.fc[k](out))
            #     out = self.dropout(out)

            # Apply a final (linear) classifier.
            out = self.sigmoid(self.graph_out(out))  # 只用Graph训练
            # out = self.sigmoid(self.sig_out(out))
        return out