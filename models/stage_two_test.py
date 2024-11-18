import torch
from torch import nn
from .blip import create_vit
import torch.nn.functional as F
from .performer import PerformerLM
import argparse
from .decorder import TransformerDecoderModel
import numpy as np


class Mymodel_two_test(nn.Module):
    def __init__(self, image_size=224, vit='base', vit_grad_ckpt=False, vit_ckpt_layer=0, embed_dim=512, momentum=0.995, use_momentum=False):
        super().__init__()

        # 初始化视觉编码器
        self.visual_encoder, vision_width = create_vit(vit, image_size, vit_grad_ckpt, vit_ckpt_layer, 0)
        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.decorder = TransformerDecoderModel()

        # Initialize momentum decoder if using momentum
        self.use_momentum = use_momentum
        if self.use_momentum:
            self.momentum_decoder = TransformerDecoderModel()
            self.momentum = momentum
            self._initialize_momentum_decoder()

    def _initialize_momentum_decoder(self):
        # Initialize momentum decoder with the same weights as the primary decoder
        for param, momentum_param in zip(self.decoder.parameters(), self.momentum_decoder.parameters()):
            momentum_param.data.copy_(param.data)
            momentum_param.requires_grad = False

    @torch.no_grad()
    def _momentum_update_decoder(self):
        # Update the momentum decoder
        for param, momentum_param in zip(self.decoder.parameters(), self.momentum_decoder.parameters()):
            momentum_param.data = momentum_param.data * self.momentum + param.data * (1.0 - self.momentum)

    def encorder_image(self, image):
        image_embeds = self.visual_encoder(image)
        image_embeds = self.vision_proj(image_embeds[:, 0, :])
        image_feat = F.normalize(image_embeds, dim=-1)
        return image_feat, image_embeds

    def forward(self, image, gene):
        if self.use_momentum:
            self._momentum_update_decoder()
        image_features, image_embed = self.encorder_image(image)
        output, loss = self.decorder(image_embed, gene)

        return output
