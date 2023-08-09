import torch
import torch.nn as nn
class MLP(nn.Module):
    def __init__(self,input_dims,middle_dims,output_dims=None):
        super(MLP,self).__init__()
        if output_dims is None:
            output_dims=input_dims
        self.model=nn.Sequential(
            nn.Linear(input_dims,middle_dims),
            torch.nn.GELU(),
            nn.Linear(middle_dims,output_dims)
        )

    def forward(self,inputs):
        return self.model(inputs)