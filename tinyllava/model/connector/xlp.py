import re

import torch.nn as nn

from . import register_connector
from .base import Connector


ACT_TYPE = {
    'relu': nn.ReLU,
    'gelu': nn.GELU
}

class CNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        act_type = 'gelu'
        self.act=ACT_TYPE[act_type]()
        
        vdim = config.vision_hidden_size
        ldim = config.hidden_size
        
        self.linear1 = nn.Linear(vdim, 4*vdim)
        self.linear2 = nn.Linear(4*vdim, 2*ldim)
        self.linear3 = nn.Linear(2*ldim, ldim)
        self.linear4 = nn.Linear(ldim, 2*ldim)
        self.linear5 = nn.Linear(2*ldim, ldim)
        

    def forward(self, x):
        x = self.act(self.linear1(x))
        x = self.act(self.linear2(x))
        x = self.act(self.linear3(x))
        res = self.act(self.linear4(x))
        res = self.linear5(res)
        return x + res
       
        
    
    
@register_connector('xlp')    
class XLPConnector(Connector):
    def __init__(self, config):
        super().__init__()
        
        self._connector = CNet(config)

   
        

    
