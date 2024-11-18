from momentum_test import MYtest
from PIL import Image
import json
from tqdm import tqdm
import anndata as ad
import numpy as np
import scanpy as sc

def get_near_genelabel(adata ,save_path, label, radius=50, genenumber=50):
    library_id = list(adata.uns["spatial"].keys())[0]
    scale = adata.uns["spatial"][library_id]["scalefactors"]["tissue_hires_scalef"]
    adata.obsm['spatial'] = adata.obsm["spatial"] * scale
    key_list = adata.obs_names.tolist()
    data_dict = {key: value.tolist() for key, value in zip(key_list, adata.obsm['spatial'])}
    center_value = np.array(data_dict[label])

    # 计算每个键到圆心的距离
    distances = {key: np.linalg.norm(np.array(value) - center_value) for key, value in data_dict.items()}

    # 找出符合条件的键
    keys_within_radius = [
        key for key, distance in distances.items()
        if distance < radius
    ]

    # 如果符合条件的键数量大于阈值
    if len(keys_within_radius) > genenumber:
        # 计算距离并排序，距离远的在前面
        sorted_keys = sorted(keys_within_radius, key=lambda k: distances[k], reverse=True)
        # 删除最远的键直到长度等于阈值
        keys_within_radius = sorted_keys[:genenumber]

    # 如果符合条件的键数量少于阈值
    elif len(keys_within_radius) < genenumber:
        # 找出所有不在 keys_within_radius 中的键
        all_keys = set(data_dict.keys())
        keys_not_in_radius = list(all_keys - set(keys_within_radius))
        # 计算距离并排序
        additional_keys = sorted(keys_not_in_radius, key=lambda k: distances[k])
        # 添加最近的键直到长度等于阈值
        keys_within_radius.extend(additional_keys[:genenumber - len(keys_within_radius)])
    keys_within_radius = [item for item in keys_within_radius if item != label]

    data_one = dict(true_label=label, false_label=keys_within_radius, gene_path=save_path)

    return data_one


adata_path = './data/Breast_cancer/GSE210616-13-2'
save_path = './data/Breast_cancer/GSE210616-13-2/'
data_list = []
adata=sc.read_visium(path=adata_path, count_file='filtered_feature_bc_matrix.h5', load_images=True)
data = json.load(open('./datasets/val_B_0.7.json', mode='r', encoding='utf-8'))
# 需要获取指定切片基因表达的名字需要修改
adata_list = [d for d in data if d.get("gene_path") == adata_path]
adata_list = [d.get("label") for d in adata_list]
for i in range(len(adata_list)):
    label = adata_list[i]
    data = get_near_genelabel(adata, save_path=save_path, label=label, radius=15, genenumber=4)
    data_list.append(data)


model = MYtest()
# 这个h5ad文件路径需要修改
adata = ad.read_h5ad('./data/Breast_cancer/GSE210616-13-2/filtered_feature_bc_matrix_one.h5ad')
data_len = len(data_list)
count = 0
for i in tqdm(data_list):
    # 图像保存路径需要修改
    image_path = './data/Breast_cancer_image_process/GSE210616-13-2/' + i['true_label'] + '.jpeg'
    image = Image.open(image_path)

    gene_label_list = i["false_label"]
    gene_label_list.insert(0, i['true_label'])

    probs = model.image_detect_gene(image=image, gene_label_list=gene_label_list, adata=adata)

    index = int(np.argmax(probs, axis=1)[0])
    if gene_label_list[index] == i['true_label']:
        count = count + 1


print('accuracy is ' + str(count / data_len))