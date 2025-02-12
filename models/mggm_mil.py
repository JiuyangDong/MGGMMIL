import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('.')
from models.mamba.mamba_ssm import SRMamba
from torch_geometric.nn import GENConv


class GCNBlock(nn.Module):
    def __init__(self, args, L, gcn_type='Mamba'):
        super(GCNBlock, self).__init__()
        self.K = args.KNN
        if '+' not in self.K:
            num_scale = 1
        else:
            num_scale = len([int(K) for K in args.KNN.split('+')])
        self.num_scale = num_scale

        self.gcn = GENConv(L, L, aggr='softmax', t=1.0, learn_t=True, num_layers=1, norm='layer')

        if gcn_type == 'Mamba':
            self.projector = SRMamba(d_model=L, d_state=16, d_conv=4, expand=1)
        else:
            raise NotImplementedError

        self.fusion = nn.Sequential(
            nn.LayerNorm(L),
            nn.Linear(L, L),
            nn.ReLU()
        )
        
    def forward(self, Feature, Edge_List):   # [1, N, L] [1, N, N]
        N = Feature.shape[1]

        MultiScaleFeatures = []

        for k in range(self.num_scale):
            NeighborFeature = self.gcn(Feature.squeeze(0), Edge_List[k].squeeze(0).permute(1, 0))
            MultiScaleFeatures.append(NeighborFeature.unsqueeze(0))

        DualScaleInformation = self.projector(torch.concat([Feature, *MultiScaleFeatures], 1))
        EachScaleInformation = sum(torch.split(DualScaleInformation, N, dim=1))

        return self.fusion(EachScaleInformation)

class GABMIL(nn.Module):
    def __init__(self, L = 512, D = 256, n_classes = 2):
        super(GABMIL, self).__init__()
        
        self.attention_V = nn.Sequential(
            nn.Linear(L,D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(L, D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(D, 1)
        
        
        self.classifier = nn.Sequential(
            nn.Linear(L, n_classes)
        )

    def forward(self, x):                       # B x N x I

        A_V = self.attention_V(x)               # B x N x D
        A_U = self.attention_U(x)               # B x N x D
        A = self.attention_weights(A_V * A_U)   # B x N x 1
        A_raw = torch.permute(A, (0, 2, 1))     # B x 1 x N
        A = F.softmax(A_raw, dim=-1)            # B x 1 x N

        M = torch.bmm(A, x)                     # B x 1 x L
        bag_logit = self.classifier(M)          # B x 1 x C

        return bag_logit.squeeze(1), A_raw, M

class MGGMMIL(nn.Module):
    def __init__(self, args=None, I=1024, L = 512, D = 256, n_classes = 2, layer=2, mil_head='GABMIL'):
        super(MGGMMIL, self).__init__()
        self.args = args
        self.layer = layer
        
        self.in_layer = nn.Sequential(
            nn.Linear(I, L),
            nn.ReLU(),
            nn.Dropout(0.25)
        )

        self.gcn_blocks = nn.ModuleList([
            GCNBlock(args, L, 'Mamba') for _ in range(layer)
        ])

        if mil_head == 'GABMIL':
            self.mil_head = GABMIL(L=L, D=D, n_classes=n_classes)
        else:
            raise NotImplementedError
            

    def forward(self, x, edge):                       # B x N x I
        assert len(x.shape) == 3

        x = self.in_layer(x)                    # B x N x L

        for i in range(self.layer):
            x = self.gcn_blocks[i](x, edge)

        bag_logit, A_raw, M = self.mil_head(x)

        return bag_logit, A_raw, M
       