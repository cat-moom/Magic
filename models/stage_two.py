import torch
from torch import nn
from .blip import create_vit
import torch.nn.functional as F
from .performer import PerformerLM
import argparse
from .decorder import TransformerDecoderModel
import numpy as np


class Mymodel_two(nn.Module):
    def __init__(self, image_size=224, vit='base', vit_grad_ckpt=False, vit_ckpt_layer=0, embed_dim=512, momentum=0.995, use_momentum=False):
        super().__init__()

        # 初始化视觉编码器
        self.visual_encoder, vision_width = create_vit(vit, image_size, vit_grad_ckpt, vit_ckpt_layer, 0)
        self.vision_proj = nn.Linear(vision_width, embed_dim)
        # 加载 checkpoint
        checkpoint = torch.load('./encoder_model/best_epoch_weights.pth')
        # 获取状态字典
        state_dict = checkpoint  # 如果 checkpoint 中直接是权重字典，则直接用 checkpoint
        # 筛选出 visual_encoder 的权重
        visual_encoder_state_dict = {k.replace('visual_encoder.', ''): v for k, v in state_dict.items() if
                                 k.startswith('visual_encoder.')}
        # 筛选出 vision_proj 的权重
        vision_proj_state_dict = {k.replace('vision_proj.', ''): v for k, v in state_dict.items() if k.startswith('vision_proj.')}
        # 加载权重到 self.visual_encoder 并检查是否加载完全
        visual_encoder_missing, visual_encoder_unexpected = self.visual_encoder.load_state_dict(
            visual_encoder_state_dict, strict=False)
        # 检查 visual_encoder 是否加载完全
        if not visual_encoder_missing and not visual_encoder_unexpected:
            print("self.visual_encoder 的所有参数加载完全。")
        else:
            print(f"self.visual_encoder 缺少参数: {visual_encoder_missing}")
            print(f"self.visual_encoder 多余参数: {visual_encoder_unexpected}")
        # 加载权重到 self.vision_proj 并检查是否加载完全
        vision_proj_missing, vision_proj_unexpected = self.vision_proj.load_state_dict(vision_proj_state_dict, strict=False)
        # 检查 vision_proj 是否加载完全
        if not vision_proj_missing and not vision_proj_unexpected:
            print("self.vision_proj 的所有参数加载完全。")
        else:
            print(f"self.vision_proj 缺少参数: {vision_proj_missing}")
            print(f"self.vision_proj 多余参数: {vision_proj_unexpected}")

        for param in self.visual_encoder.parameters():
            param.requires_grad = False
        for param in self.vision_proj.parameters():
            param.requires_grad = False

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

        return output, loss
