import cv2
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
import scanpy as sc
import anndata as ad
from .utils import cvtColor, preprocess_input
from .utils_aug import CenterCrop, ImageNetPolicy, RandomResizedCrop, Resize

class MyDataset(data.Dataset):
    def __init__(self, input_shape, lines, gene_number, random, autoaugment_flag=True):
        self.input_shape = input_shape  # 输入图像大小
        self.lines = lines  # 对应json文件内容，list格式

        self.random = random
        self.gene_number = gene_number  # 选择预测多少个高表达基因

        self.image = []
        self.label = []
        self.gene_path = []

        self.adata = {}


        for img_id, ann in enumerate(self.lines):
            self.image.append(ann['image'])
            self.label.append(ann['label'])
            self.gene_path.append(ann['gene_path'])

        self.autoaugment_flag = autoaugment_flag
        if self.autoaugment_flag:
            self.resize_crop = RandomResizedCrop(input_shape)
            self.policy = ImageNetPolicy()

            self.resize = Resize(input_shape[0] if input_shape[0] == input_shape[1] else input_shape)
            self.center_crop = CenterCrop(input_shape)

    def __len__(self):
        return len(self.lines)

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def __getitem__(self, index):
        image_path = self.image[index]
        gene_path = self.gene_path[index]
        label = self.label[index]  # 要通过细胞标签找到对应在h5文件中的基因表达

        # 获取输入到模型中的图像
        image = Image.open(image_path)
        image = cvtColor(image)
        if self.autoaugment_flag:
            image = self.AutoAugment(image, random=self.random)
        else:
            image = self.get_random_data(image, self.input_shape, random=self.random)
        image = np.transpose(preprocess_input(np.array(image, dtype='float32')), (2, 0, 1))

        # 获取输入到模型中的基因表达
        # adata = sc.read_visium(path = gene_path, count_file='filtered_feature_bc_matrix.h5', load_images=True)
        # 获取h5文件
#         if gene_path not in self.adata:
#             self.adata[gene_path] = sc.read_visium(path=gene_path, count_file='filtered_feature_bc_matrix.h5',
#                                                    load_images=True)
#         adata = self.adata[gene_path]
        # 获取h5ad文件
        if gene_path not in self.adata:
            self.adata[gene_path] = ad.read_h5ad(gene_path + '/filtered_feature_bc_matrix_decoder.h5ad')
        adata = self.adata[gene_path]

        label_index = adata.obs_names.get_loc(label)  # 提取该行的数据
        full_seq = adata.X[label_index].toarray().flatten()
        # full_seq[full_seq > 3] = 3
        full_seq = torch.from_numpy(full_seq).long()
        gene = full_seq.unsqueeze(0)


        return image,  gene

    def get_random_data(self, image, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.3, random=True):
        # ------------------------------#
        #   获得图像的高宽与目标高宽
        # ------------------------------#
        iw, ih = image.size
        h, w = input_shape

        if not random:
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)
            dx = (w - nw) // 2
            dy = (h - nh) // 2

            # ---------------------------------#
            #   将图像多余的部分加上灰条
            # ---------------------------------#
            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image, np.float32)

            return image_data

        # ------------------------------------------#
        #   对图像进行缩放并且进行长和宽的扭曲
        # ------------------------------------------#
        new_ar = iw / ih * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        scale = self.rand(0.75, 1.5)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)

        # ------------------------------------------#
        #   将图像多余的部分加上灰条
        # ------------------------------------------#
        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image

        # ------------------------------------------#
        #   翻转图像
        # ------------------------------------------#
        flip = self.rand() < .5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

        rotate = self.rand() < .5
        if rotate:
            angle = np.random.randint(-15, 15)
            a, b = w / 2, h / 2
            M = cv2.getRotationMatrix2D((a, b), angle, 1)
            image = cv2.warpAffine(np.array(image), M, (w, h), borderValue=[128, 128, 128])

        image_data = np.array(image, np.uint8)
        # ---------------------------------#
        #   对图像进行色域变换
        #   计算色域变换的参数
        # ---------------------------------#
        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        # ---------------------------------#
        #   将图像转到HSV上
        # ---------------------------------#
        hue, sat, val = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype = image_data.dtype
        # ---------------------------------#
        #   应用变换
        # ---------------------------------#
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)
        return image_data

    def AutoAugment(self, image, random=True):
        if not random:
            image = self.resize(image)
            image = self.center_crop(image)
            return image

        # ------------------------------------------#
        #   resize并且随即裁剪
        # ------------------------------------------#
        image = self.resize_crop(image)

        # ------------------------------------------#
        #   翻转图像
        # ------------------------------------------#
        flip = self.rand() < .5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # ------------------------------------------#
        #   随机增强
        # ------------------------------------------#
        image = self.policy(image)
        return image


def dataset_collate(batch):
    images = []
    genes = []
    for image, gene in batch:
        images.append(image)
        genes.append(gene)

    images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    genes = torch.stack(genes).squeeze(dim=1)
    return images, genes
