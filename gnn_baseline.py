import pandas as pd 
from utils import cprint, evaluationAUC_F1

def load_data(prefix='train'):
    item_table = pd.read_csv('task1_data/{}_item_feature_table.csv'.format(prefix))
    user_table = pd.read_csv('task1_data/{}_user_feature_table.csv'.format(prefix))
    print('user feature shape:', user_table.shape)
    print('item feature shape:', item_table.shape)
    
    edge_table = pd.read_csv('task1_data/{}_e.csv'.format(prefix))
    return item_table, user_table, edge_table

train_item_table, train_user_table, train_edge_table = load_data('train')
test_item_table, test_user_table, test_edge_table = load_data('test')


import dgl
import torch 
def load_dgl_graph(edge_table, user_table, item_table):
    graph_data = {
    ('user', 'buy', 'item'): (edge_table['userid'].values, edge_table['itemid'].values),
    ('item', 'buyed', 'user'): (edge_table['itemid'].values, edge_table['userid'].values), 
    }
    g = dgl.heterograph(graph_data)
    print(g)
    print(g.ntypes, g.etypes, g.canonical_etypes)
    g.nodes['user'].data['feature'] = torch.from_numpy(user_table[user_table.columns[1:]].values).float()
    g.nodes['item'].data['feature'] = torch.from_numpy(item_table[item_table.columns[1:]].values).float()
    g.edges['buy'].data['label'] = torch.from_numpy(edge_table['label'].values).long()
    return g

train_g = load_dgl_graph(train_edge_table, train_user_table, train_item_table)
test_g = load_dgl_graph(test_edge_table, test_user_table, test_item_table)

class Model(torch.nn.Module):
    def __init__(self, user_feature=32, item_feature=43, hidden_feature=64):
        super(Model, self).__init__()
        self.conv = dglnn.HeteroGraphConv({
            'buy' : dglnn.GraphConv(user_feature, hidden_feature, norm='none', weight=True, bias=True),
            'buyed' : dglnn.GraphConv(item_feature, hidden_feature, norm='none', weight=True, bias=True)},
        aggregate='sum')

        self.cls = torch.nn.Sequential(
            torch.nn.Linear(hidden_feature*2+user_feature+item_feature, hidden_feature),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_feature, hidden_feature),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_feature, 1),
            torch.nn.Sigmoid()
        )
        self.bceloss = torch.nn.BCELoss()

    def apply_edges(self, edges):
        """
        Computes a scalar score for each edge of the given graph.

        Parameters
        ----------
        edges :
            Has three members ``src``, ``dst`` and ``h1``, each of
            which is a dictionary representing the features of the
            source nodes, the destination nodes, and the edges
            themselves.

        Returns
        -------
        dict
            A dictionary of new edge features.
        """
        h = torch.cat([edges.src['feature'], edges.dst['feature'], edges.src['h1'], edges.dst['h1']], 1)
        return {'score': self.cls(h)}

    def forward(self, g, x):
        h = self.conv(g, x)
        g.nodes['user'].data['h1'] = h['user']
        g.nodes['item'].data['h1'] = h['item']
        # print(g.edges['buy'])
       
        g.apply_edges(self.apply_edges, etype='buy')
        
        scores = g.edges['buy'].data['score'].squeeze()
        labels = g.edges['buy'].data['label'].float()
        # print(scores.shape, labels.shape)
        return scores, labels

import dgl.nn.pytorch as dglnn
from torch.nn import functional as F
device = 'cuda:0'
sage1 = Model(hidden_feature=256).to(device)
opt = torch.optim.SGD(sage1.parameters(), lr=1e-2, weight_decay=1e-4)
for epoch in range(500):
    scores, labels = sage1(train_g.to(device), {'user':train_g.nodes['user'].data['feature'].to(device), 'item':train_g.nodes['item'].data['feature'].to(device)})
    loss = F.binary_cross_entropy(scores, labels)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if epoch % 50 == 0:
        print("TRAIN epochs {} loss: {:.2f}".format(epoch, loss.item()))
    
scores, labels = sage1(test_g.to(device), {'user':test_g.nodes['user'].data['feature'].to(device), 'item':test_g.nodes['item'].data['feature'].to(device)})
scores, labels = scores.cpu().detach().numpy(), labels.cpu().detach().numpy()
evaluationAUC_F1(scores, labels)