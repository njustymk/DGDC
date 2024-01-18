import pdb
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

class DGDC(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.FTE = nn.Sequential(
            nn.Linear(args.vector_len, args.vector_len),
            nn.ReLU(inplace=True),
            nn.Linear(args.vector_len, args.vector_len),
        )
        print(self.FTE)
        self.MLP1 = nn.Sequential(
            nn.Linear(args.vector_len, args.vector_len),
            nn.ReLU(inplace=True),
            nn.Linear(args.vector_len, args.vector_len),
        )
        print(self.MLP1)
        self.MLP2 = nn.Sequential(
            Rearrange('b c n d -> b c d n'),
            nn.Linear(args.len_old, args.len_old),
            nn.ReLU(inplace=True),
            nn.Linear(args.len_old, args.len_old),
            Rearrange('b c d n -> b c n d')
        )
        print(self.MLP2)
        self.Head = nn.Sequential(
            nn.Linear(args.vector_len*2, args.num_classes),
        )
        print(self.Head)

    def forward(self, gs, x):
        
        gs = gs.transpose(0,1)
        x = self.FTE(x)-self.FTE(gs)
        x = x.transpose(0,1)
        x = x.view(1, x.size(0), x.size(1), x.size(2))
        x1 = self.MLP1(x)
        x2 = self.MLP2(x)
        x = torch.cat((x1, x2), 3)
        x = x.view(1, x.size(-2), x.size(-1))
        x = x.mean(dim=1)
        x = self.Head(x)
        return x


        # gs = gs.transpose(0,1)
        # diff = self.FTE(x)-self.FTE(gs)
        # diff = diff.transpose(0,1)
        # diff = diff.view(1, diff.size(0), diff.size(1), diff.size(2))

        # mlp1_out = self.MLP1(diff)
        # mlp2_out = self.MLP2(diff)
        # mlp_out = torch.cat((mlp1_out, mlp2_out), 3)
        # mlp_out = mlp_out.view(1, mlp_out.size(-2), mlp_out.size(-1))
        # mlp_out = mlp_out.mean(dim=1)
        # res = self.Head(mlp_out)
        # return diff, mlp1_out, mlp2_out, mlp_out, res
              