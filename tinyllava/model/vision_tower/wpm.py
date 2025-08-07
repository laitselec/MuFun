import torch
from . import register_vision_tower
from .base import VisionTower
from transformers import AutoProcessor
import os
from torch import nn

@register_vision_tower('wpm')      
class WpmAudioTower(VisionTower):
    def __init__(self, cfg):
        super().__init__(cfg)
        # self.enc_model_path="w3_enc_fp32.bin"
        # self._vision_tower = torch.load(self.enc_model_path,weights_only=False)
        self._vision_tower = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3").get_encoder()
        self._image_processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")
        self.pool_stride = 5
        self.avg_pooler = nn.AvgPool1d(self.pool_stride, stride=self.pool_stride)
        self.features_layers = [0, 7, 15, 32]
        
    def _load_model(self, vision_tower_name, **kwargs):
        pretrained_vision_tower_path = kwargs.pop('pretrained_vision_tower_path', None)
        if pretrained_vision_tower_path is None:
            # model_name_or_path_dinov2 = kwargs.pop('model_name_or_path2')
            # self._vision_tower.clip = self._vision_tower.clip.from_pretrained(vision_tower_name, **kwargs)
            # self._vision_tower.dinov2 = self._vision_tower.dinov2.from_pretrained(model_name_or_path_dinov2, **kwargs)
            
            print("Loading vision tower1 from ", vision_tower_name)
        else: # nn.Module
            if pretrained_vision_tower_path is not None:
                vision_tower_weights = torch.load(os.path.join(pretrained_vision_tower_path, 'pytorch_model.bin'), map_location='cpu')
                def get_w(weights, keyword):
                    return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
                self._vision_tower.load_state_dict(vision_tower_weights)
            print("Loading vision tower from ", pretrained_vision_tower_path)

        
        
    def forward(self, x, **kwargs):
    #    x=x.to(self._vision_tower.dtype)
       if len(x.shape)==4:
        #    x=torch.squeeze(x, 0)
           x=torch.squeeze(x, 1)
    #    image_features = self._vision_tower(x, output_hidden_states=False)
    #    hidden_states = image_features.last_hidden_state
       image_features = self._vision_tower(x, output_hidden_states=True).hidden_states
       hidden_states = torch.cat([image_features[il] for il in self.features_layers], dim=-1)

       hidden_states = hidden_states.permute(0, 2, 1)
       hidden_states = self.avg_pooler(hidden_states)
       hidden_states = hidden_states.permute(0, 2, 1)

       return hidden_states
