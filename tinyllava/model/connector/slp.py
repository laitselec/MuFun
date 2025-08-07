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
        self.linear2 = nn.Linear(4*vdim, ldim)
        

    def forward(self, x):
        x = self.act(self.linear1(x))
        return self.linear2(x)
        
        
    
    
@register_connector('slp')    
class SLPConnector(Connector):
    def __init__(self, config):
        super().__init__()
        
        self._connector = CNet(config)

   
        

    
