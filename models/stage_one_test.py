import torch
from torch import nn
from .blip import create_vit
import torch.nn.functional as F
from .performer import PerformerLM
import argparse
from .crossattention import StackedModel
import numpy as np


class Mymodel(nn.Module):
    def __init__(self, image_size=224, vit='base', vit_grad_ckpt=False, vit_ckpt_layer=0, embed_dim=512,momentum=0.995):
        super().__init__()

        parser = argparse.ArgumentParser()
        parser.add_argument("--bin_num", type=int, default=5, help='Number of bins.')
        parser.add_argument("--gene_num", type=int, default=16906, help='Number of genes.')
        parser.add_argument("--pos_embed", type=bool, default=True, help='Using Gene2vec encoding or not.')
        args = parser.parse_args()
        CLASS = args.bin_num + 2
        SEQ_LEN = args.gene_num + 1
        POS_EMBED_USING = args.pos_embed

        # Initialize the visual encoder
        self.visual_encoder, vision_width = create_vit(vit, image_size, vit_grad_ckpt, vit_ckpt_layer, 0)
        # self.cross_block = StackedModel(input_dim1=512, input_dim2=512, hidden_dim=1024, n_heads=8,n_stacks=6)
        self.cross_block = StackedModel(input_dim=512, hidden_dim=512, forward_dim=1024, n_stacks=6, n_heads=8)

        # Load pre-trained weights
        if vit == 'base':
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
                map_location="cpu", check_hash=True)
            state_dict = checkpoint["model"]
            msg = self.visual_encoder.load_state_dict(state_dict, strict=False)
        elif vit == 'large':
            from timm.models.helpers import load_custom_pretrained
            from timm.models.vision_transformer import default_cfgs
            load_custom_pretrained(self.visual_encoder, default_cfgs['vit_large_patch16_224_in21k'])

        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.itm_head = nn.Linear(512, 2)

        # Initialize the gene model
        self.gene = PerformerLM(
            num_tokens=CLASS,
            dim=200,
            depth=6,
            max_seq_len=SEQ_LEN,
            heads=10,
            local_attn_heads=0,
            g2v_position_emb=POS_EMBED_USING
        )
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gene.load_state_dict(torch.load('./panglao_pretrain.pth', map_location=device), strict=False)
        for param in self.gene.parameters():
            param.requires_grad = False
        for param in self.gene.norm.parameters():
            param.requires_grad = True
        for param in self.gene.performer.net.layers[-2].parameters():
            param.requires_grad = True
        for param in self.gene.output.parameters():
            param.requires_grad = True

        # Initialize momentum encoder for visual and gene encoders
        self.momentum_visual_encoder = create_vit(vit, image_size, vit_grad_ckpt, vit_ckpt_layer, 0)[0]
        self.momentum_gene = PerformerLM(
            num_tokens=CLASS,
            dim=200,
            depth=6,
            max_seq_len=SEQ_LEN,
            heads=10,
            local_attn_heads=0,
            g2v_position_emb=POS_EMBED_USING
        )

        # Copy initial weights
        self._initialize_momentum_encoder()

        self.momentum = momentum  # Momentum factor
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def _initialize_momentum_encoder(self):
        # Initialize momentum encoders with the same weights as the primary encoders
        for param, momentum_param in zip(self.visual_encoder.parameters(), self.momentum_visual_encoder.parameters()):
            momentum_param.data.copy_(param.data)
            momentum_param.requires_grad = False  # Momentum encoder does not require gradient

        for param, momentum_param in zip(self.gene.parameters(), self.momentum_gene.parameters()):
            momentum_param.data.copy_(param.data)
            momentum_param.requires_grad = False

    @torch.no_grad()
    def _momentum_update_encoder(self):
        # Update momentum encoders
        for param, momentum_param in zip(self.visual_encoder.parameters(), self.momentum_visual_encoder.parameters()):
            momentum_param.data = momentum_param.data * self.momentum + param.data * (1.0 - self.momentum)

        for param, momentum_param in zip(self.gene.parameters(), self.momentum_gene.parameters()):
            momentum_param.data = momentum_param.data * self.momentum + param.data * (1.0 - self.momentum)

    def encorder_image(self, image):
        image_embeds = self.visual_encoder(image)
        image_embeds = self.vision_proj(image_embeds[:, 0, :])
        image_feat = F.normalize(image_embeds, dim=-1)
        return image_feat, image_embeds

    def encorder_gene(self, gene):
        return self.gene(gene)

    def forward(self, image, gene):
        # Update the momentum encoder
        self._momentum_update_encoder()

        # Encode the image and gene using the primary encoders
        image_features, image_embeds = self.encorder_image(image)
        gene_embeds = self.encorder_gene(gene)

        # Normalize gene features
        gene_features = gene_embeds / gene_embeds.norm(dim=-1, keepdim=True)

        # Compute logits
        logit_scale = self.logit_scale.exp()
        sim_i2g = image_features @ gene_features.t()
        logits_per_image_gene = logit_scale * sim_i2g

        # -------------------------------------------------------------------------------
        bs = image_embeds.size(0)
        sim_g2i = sim_i2g.t()
        output_pos = self.cross_block(gene_embeds, image_embeds)


        with torch.no_grad():
            # Compute similarity and softmax weights
            weights_g2i = F.softmax(sim_g2i[:, :bs], dim=1) + 1e-4
            weights_g2i.fill_diagonal_(0)
            weights_i2g = F.softmax(sim_i2g[:, :bs], dim=1) + 1e-4
            weights_i2g.fill_diagonal_(0)

        # Select negative images for each gene
        image_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_g2i[b], 1).item()
            image_embeds_neg.append(image_embeds[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg, dim=0)

        # # Select negative gene for each image
        gene_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2g[b], 1).item()
            gene_embeds_neg.append(gene_embeds[neg_idx])
        gene_embeds_neg = torch.stack(gene_embeds_neg, dim=0)

        gene_embeds_all = torch.cat([gene_embeds, gene_embeds_neg], dim=0)
        image_embeds_all = torch.cat([image_embeds_neg, image_embeds], dim=0)

        output_neg = self.cross_block(gene_embeds_all, image_embeds_all)

        vl_embeddings = torch.cat([output_pos.squeeze(0), output_neg.squeeze(0)], dim=0)
        vl_output = self.itm_head(vl_embeddings)

        itm_labels = torch.cat([torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)], dim=0).to(image.device)
        loss_gtm = F.cross_entropy(vl_output, itm_labels)

        return logits_per_image_gene, loss_gtm



class Mymodel_test(nn.Module):
    def __init__(self, image_size=224, vit='base', vit_grad_ckpt=False, vit_ckpt_layer=0, embed_dim=512, momentum=0.995, use_momentum=False):
        super().__init__()

        parser = argparse.ArgumentParser()
        parser.add_argument("--bin_num", type=int, default=5, help='Number of bins.')
        parser.add_argument("--gene_num", type=int, default=16906, help='Number of genes.')
        parser.add_argument("--pos_embed", type=bool, default=True, help='Using Gene2vec encoding or not.')
        args = parser.parse_args()
        CLASS = args.bin_num + 2
        SEQ_LEN = args.gene_num + 1
        POS_EMBED_USING = args.pos_embed

        # 初始化视觉编码器
        self.visual_encoder, vision_width = create_vit(vit, image_size, vit_grad_ckpt, vit_ckpt_layer, 0)
        self.cross_block = StackedModel(input_dim=512, hidden_dim=512, forward_dim=1024, n_stacks=6, n_heads=8)

        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.itm_head = nn.Linear(512, 2)

        # 初始化基因模型
        self.gene = PerformerLM(
            num_tokens=CLASS,
            dim=200,
            depth=6,
            max_seq_len=SEQ_LEN,
            heads=10,
            local_attn_heads=0,
            g2v_position_emb=POS_EMBED_USING
        )

        # 动量编码器设置
        self.use_momentum = use_momentum
        if self.use_momentum:
            self.momentum_visual_encoder = create_vit(vit, image_size, vit_grad_ckpt, vit_ckpt_layer, 0)[0]
            self.momentum_gene = PerformerLM(
                num_tokens=CLASS,
                dim=200,
                depth=6,
                max_seq_len=SEQ_LEN,
                heads=10,
                local_attn_heads=0,
                g2v_position_emb=POS_EMBED_USING
            )
            self._initialize_momentum_encoder()

        self.momentum = momentum  # 动量因子

    def _initialize_momentum_encoder(self):
        # 初始化动量编码器，使其与主编码器权重相同
        for param, momentum_param in zip(self.visual_encoder.parameters(), self.momentum_visual_encoder.parameters()):
            momentum_param.data.copy_(param.data)
            momentum_param.requires_grad = False  # 动量编码器不需要梯度

        for param, momentum_param in zip(self.gene.parameters(), self.momentum_gene.parameters()):
            momentum_param.data.copy_(param.data)
            momentum_param.requires_grad = False

    @torch.no_grad()
    def _momentum_update_encoder(self):
        # 更新动量编码器
        for param, momentum_param in zip(self.visual_encoder.parameters(), self.momentum_visual_encoder.parameters()):
            momentum_param.data = momentum_param.data * self.momentum + param.data * (1.0 - self.momentum)

        for param, momentum_param in zip(self.gene.parameters(), self.momentum_gene.parameters()):
            momentum_param.data = momentum_param.data * self.momentum + param.data * (1.0 - self.momentum)

    def encorder_image(self, image):
        image_embeds = self.visual_encoder(image)
        image_embeds = self.vision_proj(image_embeds[:, 0, :])
        image_feat = F.normalize(image_embeds, dim=-1)
        return image_feat, image_embeds

    def encorder_gene(self, gene):
        return self.gene(gene)

    def predict_match(self, image_embed, gene_embeds):
        # 计算正样本（索引0是正确匹配的基因表达）和图片之间的交互
        output_pos = self.cross_block(gene_embeds[0].unsqueeze(0), image_embed)
        # 初始化一个列表来存储负样本的输出
        output_neg_list = []
        # 遍历其余4个负样本基因表达（索引1到4）
        for i in range(1, gene_embeds.size(0)):
            output_neg = self.cross_block(gene_embeds[i].unsqueeze(0), image_embed)
            output_neg_list.append(output_neg)
        # 将正样本和负样本的输出连接起来
        output_all = torch.cat([output_pos] + output_neg_list, dim=0)
        # 通过itm_head（图像-文本匹配头）
        vl_output = self.itm_head(output_all.squeeze(0))

        # itm_head的输出给出每个基因-图片对的得分
        # 第一个得分（索引0）对应正确的基因-图片对
        probabilities = F.softmax(vl_output, dim=-1)
        # index = torch.argmax(probabilities[:,1])
        index = probabilities[:, 1]
        score_correct_pair = probabilities[0][1]
        return score_correct_pair, index


    def forward(self, image, gene):
        # 如果需要，更新动量编码器
        if self.use_momentum:
            self._momentum_update_encoder()

        # 使用主编码器对图像和基因进行编码
        image_features, image_embed = self.encorder_image(image)
        gene_embeds = self.encorder_gene(gene)

        # 归一化基因特征
        gene_features = gene_embeds / gene_embeds.norm(dim=-1, keepdim=True)

        # 计算logits
        logit_scale = self.logit_scale.exp()
        logits_per_image_gene = logit_scale * image_features @ gene_features.t()


        return logits_per_image_gene
