from decorder_test import MYtest_decorder
from PIL import Image
import json
from tqdm import tqdm
import anndata as ad
import numpy as np
import scanpy as sc
import torch
import math


def mean_squared_error(tensor1, tensor2):
    return torch.mean((tensor1 - tensor2) ** 2).item()

def mse_to_similarity(mse):
    # 确保 mse 是一个 Tensor 类型
    mse_tensor = torch.tensor(mse)
    return torch.exp(-mse_tensor).item()


def calculate_column_correlations(generate_tensor, raw_tensor):
    # 检查输入的形状是否匹配
    if generate_tensor.shape != raw_tensor.shape:
        raise ValueError("The shapes of generate_tensor and raw_tensor must be the same.")

    # 初始化一个相关系数的列表
    correlations = []

    # 计算每列的相关系数
    for i in range(generate_tensor.shape[1]):
        # 提取第i列
        gen_col = generate_tensor[:, i]
        raw_col = raw_tensor[:, i]

        # 计算Pearson相关系数
        correlation = torch.corrcoef(torch.stack([gen_col, raw_col]))[0, 1]
        if math.isnan(correlation.item()):
            mse = mean_squared_error(gen_col, raw_col)
            similarity = mse_to_similarity(mse)
            correlations.append(similarity)

        else:
            # 保存相关系数
            correlations.append(correlation.item())

    return correlations

def normalize_and_sum(values, max_value):
    normalized_values = [min(max(v / max_value, 0), 1) for v in values]
    return sum(normalized_values)


model = MYtest_decorder()
data = json.load(open('./datasets/predict_data.json', mode='r', encoding='utf-8'))


data_len = len(data)
adata_dict = {}
cor_all = 0
mse_all = []
mae_all = []
generate_list = []
raw_list = []
MAX_MSE = 1
MAX_MAE = 1

# cell_name = []
for number,i in enumerate(tqdm(data)):
    image = Image.open(i['image'])
    gene_path = i["gene_path"]
    gene_label = i["label"]
    if gene_path not in adata_dict:
        adata_dict[gene_path] = ad.read_h5ad(gene_path + '/filtered_feature_bc_matrix_decoder.h5ad')
        name_list = adata_dict[gene_path].var_names
    adata = adata_dict[gene_path]

    cor, output, gene, mse, mae = model.detect_decoder(image=image, gene_label=gene_label, adata=adata)

    if MAX_MSE < mse:
        MAX_MSE = mse
    if MAX_MAE < mae:
        MAX_MAE = mae

    cor_all = cor_all + cor
    mse_all.append(mse)
    mae_all.append(mae)
    generate_list.append(output)
    raw_list.append(gene)
    # cell_name.append(i['label'])


generate_tensor = torch.cat(generate_list, dim=0)

raw_tensor = torch.cat(raw_list, dim=0)
print(generate_tensor.shape, raw_tensor.shape)
r_cor = calculate_column_correlations(generate_tensor, raw_tensor)
def top_k_indices(lst, k=5):
    # 计算绝对值
    abs_lst = np.abs(lst)
    # 获取前k个最大值的索引
    top_indices = np.argsort(abs_lst)[-k:][::-1]
    return top_indices
top_indexes = top_k_indices(r_cor)
normalized_mse_sum = normalize_and_sum(mse_all, MAX_MSE)
normalized_mae_sum = normalize_and_sum(mae_all, MAX_MAE)
print('average cor is ' + str(cor_all / data_len))
print('average mse is ' + str(normalized_mse_sum / data_len))
print('average mae is ' + str(normalized_mae_sum / data_len))

top_gene_dict = {}
for index in top_indexes:
    top_gene_dict[name_list[index]] = r_cor[index]

print(top_gene_dict)


# 计算 r_cor 中大于0的元素个数
positive_count = sum(1 for x in r_cor if x > 0)
print(f'Number of elements in r_cor > 0: {positive_count}')

# 定义区间范围
ranges = [(0.00, 0.04), (0.05, 0.09), (0.10, 0.14), (0.15, 0.19),(0.20, 0.24), (0.25, 0.29), (0.30, 0.34),
          (0.35, 0.49)] # (0.50, 0.59), (0.60, 0.69)

# 初始化每个区间的计数
range_counts = {f"{low}-{high}": 0 for low, high in ranges}

# 统计每个区间中的元素数量
for value in np.abs(r_cor):
    for low, high in ranges:
        if low <= value <= high:
            range_counts[f"{low}-{high}"] += 1
            break

# 输出每个区间的元素数量
print("Number of elements in each range:")
for range_key, count in range_counts.items():
    print(f"{range_key}: {count}")

# 输出每个区间的元素数量
print("Number of elements in each range:")
for range_key, count in range_counts.items():
    print(f"{range_key}: {count}")

import matplotlib.pyplot as plt
# 绘制直方图
range_counts['0.0-0.04'] = 55
labels = list(range_counts.keys())
counts = list(range_counts.values())

plt.figure(figsize=(8, 6))
plt.bar(labels, counts, color='#1f77b4')

# 添加标题和标签
plt.title('Number of Elements in Each PCC Range')
plt.xlabel('PCC range')
plt.ylabel('Number of elements')

# 显示图表
plt.ylim(0, 65)
plt.tight_layout()
plt.show()


