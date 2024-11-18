import numpy as np
import torch
from torch import nn
from models.stage_one_test import Mymodel_test
from utils.utils import (cvtColor,  letterbox_image,
                         preprocess_input)

class MYtest_momentum(object):
    _defaults = {
        "model_path": './encoder_model/best_epoch_weights.pth',
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

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.generate()

    def generate(self):

        self.net    = Mymodel_test(use_momentum=False)
        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device), strict=False)
        self.net    = self.net.eval()
        print('{} model loaded.'.format(self.model_path))

        if self.cuda:
            self.net = self.net.cuda()


    def image_detect_gene(self, image, gene_label_list, adata, CLASS=5):
        image = cvtColor(image)
        image_data = letterbox_image(image, [224, 224], self.letterbox_image)
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)
        genes = []
        for label in gene_label_list:
            label_index = adata.obs_names.get_loc(label)  # 提取该行的数据
            full_seq = adata.X[label_index].toarray().flatten()
            full_seq[full_seq > (CLASS - 2)] = CLASS - 2
            full_seq = torch.from_numpy(full_seq).long()
            gene = full_seq.unsqueeze(0)
            genes.append(gene)

        genes = torch.cat(genes, dim=0)
        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
                genes = genes.cuda()

            gene_embedding = self.net.encorder_gene(genes)
            image_embedding, image_e = self.net.encorder_image(images)
            score, max_index = self.net.predict_match(image_e, gene_embedding)

            image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)
            gene_embedding = gene_embedding / gene_embedding.norm(dim=-1, keepdim=True)

            logit_scale = self.logit_scale.exp()

            logits_per_image_gene = logit_scale * image_embedding @ gene_embedding.t()
            probs = logits_per_image_gene.softmax(dim=-1).cpu().numpy()

        return probs, score, max_index

    def test_all_slice_image(self, image, genes):
        image = cvtColor(image)
        image_data = letterbox_image(image, [224, 224], self.letterbox_image)
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
                genes = genes.cuda()

            gene_embedding = self.net.encorder_gene(genes)
            image_embedding, image_e = self.net.encorder_image(images)
            score, max_index = self.net.predict_match(image_e, gene_embedding)

            image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)
            gene_embedding = gene_embedding / gene_embedding.norm(dim=-1, keepdim=True)

            logit_scale = self.logit_scale.exp()

            logits_per_image_gene = logit_scale * image_embedding @ gene_embedding.t()
            probs = logits_per_image_gene.softmax(dim=-1).cpu().numpy()

        return probs, score, max_index



if __name__ == 'main':
    test = MYtest_momentum()
