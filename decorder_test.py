import numpy as np
import torch
from torch import nn
from models.stage_two_test import Mymodel_two_test
from models.stage_one_test import Mymodel_test
from utils.utils import (cvtColor,  letterbox_image,
                         preprocess_input)


class MYtest_decorder(object):
    _defaults = {
        "model_path": './decoder_model/best_epoch_weights.pth',
        "letterbox_image": False,
        "cuda": True,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)

        self.generate()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def generate(self):

        self.decoder_net = Mymodel_two_test(use_momentum=False)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.decoder_net.load_state_dict(torch.load(self.model_path, map_location=device), strict=False)
        self.decoder_net = self.decoder_net.eval()
        print('{} decoder model loaded.'.format(self.model_path))


        if self.cuda:
            self.decoder_net = self.decoder_net.to(device)

    def detect_decoder(self, image, gene_label, adata):
        def gene_process(gene_label=None, adata=None, CLASS=5, genes=False, output=None):
            if genes:
                full_seq = output.cpu().numpy().flatten()
                full_seq = full_seq.astype(np.int64).astype(np.float64)
                full_seq[full_seq < 0] = 0
            else:
                label_index = adata.obs_names.get_loc(gene_label)  # 提取该行的数据
                full_seq = adata.X[label_index].toarray().flatten()
            if CLASS:
                full_seq[full_seq > (CLASS - 2)] = CLASS - 2
            full_seq = torch.from_numpy(full_seq).long()
            gene = full_seq.unsqueeze(0)
            return gene

        def calculate_correlation(output, gene):
            # Flatten the tensors and clamp negative values to zero
            output_flat = torch.clamp(output.flatten(), min=0).float()  # 确保是浮点型
            gene_flat = gene.flatten().float()  # 确保是浮点型

            # Calculate the Pearson correlation coefficient
            output_mean = torch.mean(output_flat)
            gene_mean = torch.mean(gene_flat)

            output_diff = output_flat - output_mean
            gene_diff = gene_flat - gene_mean

            numerator = torch.sum(output_diff * gene_diff)
            denominator = torch.sqrt(torch.sum(output_diff ** 2) * torch.sum(gene_diff ** 2))

            correlation = numerator / denominator

            return correlation.item()


        def calculate_mse_and_mae(output, gene):
            output = output.float()
            gene = gene.float()

            # 计算均方误差 (MSE)
            mse = torch.mean((output - gene) ** 2)

            # 计算平均绝对误差 (MAE)
            mae = torch.mean(torch.abs(output - gene))

            return mse.item(), mae.item()

        image = cvtColor(image)
        image_data = letterbox_image(image, [224, 224], self.letterbox_image)
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)
        gene = gene_process(gene_label=gene_label, adata=adata, CLASS=False)
        with torch.no_grad():
            image = torch.from_numpy(image_data)
            if self.cuda:
                image = image.cuda()
                gene = gene.cuda()
            output = self.decoder_net(image, gene)
            # 将output中所有小于0的数据变成0
            output = torch.clamp(output, min=0)
            # output = torch.abs(output)

            mse, mae = calculate_mse_and_mae(output, gene)
            x = calculate_correlation(output, gene)

        return x, output, gene, mse, mae


    def tcga_decoder(self, image, gene):
        image = cvtColor(image)
        image_data = letterbox_image(image, [224, 224], self.letterbox_image)
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)
        gene = np.array(gene)
        full_seq = torch.from_numpy(gene).long()
        gene = full_seq.unsqueeze(0)
        with torch.no_grad():
            image = torch.from_numpy(image_data)
            if self.cuda:
                image = image.cuda()
                gene = gene.cuda()
            output = self.decoder_net(image, gene)
            # 将output中所有小于0的数据变成0
            output = torch.clamp(output, min=0)

        return output
