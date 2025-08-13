import re

import torch.nn as nn

from . import register_connector
from .base import Connector


ACT_TYPE = {
    'relu': nn.ReLU,
    'gelu': nn.GELU
}
def extract_numbers(s):
    match = re.findall(r'(\d+)[ix]', s)
    if len(match) == 2:
        return tuple(map(int, match))
    return None

class CNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        ix, hx = extract_numbers(config.connector_type)
        act_type = 'gelu'
        self.act=ACT_TYPE[act_type]()
        
        vdim = config.vision_hidden_size*ix
        ldim = config.hidden_size
        
        self.linear1 = nn.Linear(vdim, hx*vdim)
        self.linear2 = nn.Linear(hx*vdim, ldim)
        

    def forward(self, x):
        x = self.act(self.linear1(x))
        return self.linear2(x)
        
        
    
    
@register_connector('blp')    
class MLPConnector(Connector):
    def __init__(self, config):
        super().__init__()
        
        self._connector = CNet(config)

