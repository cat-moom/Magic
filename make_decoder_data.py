import os
import numpy as np
import pandas as pd
import scanpy as sc

path = './data/Breast_cancer/'
file_path = os.listdir(path)

# 读取基因名替换文件
replacement_file = 'datasets/features.tsv'
replacement_df = pd.read_csv(replacement_file, sep='\t', header=None, usecols=[0, 1], names=['Original', 'Replacement'])
replacement_dict = dict(zip(replacement_df['Original'], replacement_df['Replacement']))

# 用于存储每个病人的基因集合
all_patient_genes = []

# 用于存储所有病人中共同基因的表达量
gene_expression_dict = {}

# Step 1: 获取所有病人的基因集合
for i in file_path:
    if i == 'NC_BRCA':
        continue

    gene_path = path + i
    print(f"Processing slice: {i}")

    # 读取 Visium 数据
    adata = sc.read_visium(path=gene_path, count_file='filtered_feature_bc_matrix.h5', load_images=True)

    # 确保基因名称是唯一的
    adata.var_names_make_unique()

    # 替换指定病人的基因名
    if i in ['GSM5420753', 'BRCA_BlockASection1_10x', 'BRCA_BlockASection2_10x',
             'BRCA_DuctalCarcinomaInSitu_InvasiveCarcinoma_10x_FFPE', 'BRCA_WholeTranscriptomeAnalysis_10x',
             'BRCA_InvasiveDuctalCarcinoma_StainedWithFluorescent_CD3Antibody_10x',
             'BRCA_Visium_FreshFrozen_WholeTranscriptome_10x']:
        adata.var_names = [replacement_dict.get(gene, gene) for gene in adata.var_names]

    # 再次确保基因名称唯一
    adata.var_names_make_unique()

    # 获取当前病人的基因名称集合
    gene_names = set(adata.var_names[:20000].tolist())  # 提取前20000个基因

    # 将基因集合加入列表中
    all_patient_genes.append(gene_names)

# 计算所有病人基因集合的交集
gene_name_intersection = set.intersection(*all_patient_genes)

print(f"Total genes in intersection: {len(gene_name_intersection)}")

# Step 2: 获取交集中基因的表达矩阵并计算总表达量
for i in file_path:
    if i == 'NC_BRCA':
        continue

    gene_path = path + i
    print(f"Processing and saving gene expressions for slice: {i}")

    # 读取 Visium 数据
    adata = sc.read_visium(path=gene_path, count_file='filtered_feature_bc_matrix.h5', load_images=True)

    # 确保基因名称唯一
    adata.var_names_make_unique()

    # 替换指定病人的基因名
    if i in ['GSM5420753', 'BRCA_BlockASection1_10x', 'BRCA_BlockASection2_10x',
             'BRCA_DuctalCarcinomaInSitu_InvasiveCarcinoma_10x_FFPE', 'BRCA_WholeTranscriptomeAnalysis_10x',
             'BRCA_InvasiveDuctalCarcinoma_StainedWithFluorescent_CD3Antibody_10x',
             'BRCA_Visium_FreshFrozen_WholeTranscriptome_10x']:
        adata.var_names = [replacement_dict.get(gene, gene) for gene in adata.var_names]

    # 再次确保基因名称唯一
    adata.var_names_make_unique()

    # 提取交集中基因的表达矩阵
    common_genes = list(gene_name_intersection)
    # 检查 common_genes 中是否有不存在于 adata.var_names 的基因
    missing_genes = [gene for gene in common_genes if gene not in adata.var_names]
    if missing_genes:
        print(f"Warning: Missing genes in {i}: {missing_genes}")
    expression_matrix = adata[:, common_genes].X.toarray()  # 提取共同基因的表达量
    total_expression = np.sum(expression_matrix, axis=0)  # 按基因求和，得到每个基因的总表达

    # 更新字典中该基因的表达值
    for idx, gene in enumerate(common_genes):
        if gene in gene_expression_dict:
            gene_expression_dict[gene].append(total_expression[idx])
        else:
            gene_expression_dict[gene] = [total_expression[idx]]

# Step 3: 计算每个基因在所有病人中的平均表达量，并选择前300个高表达基因
average_expression = {gene: np.mean(expr) for gene, expr in gene_expression_dict.items()}

# 将平均表达量转换为DataFrame，方便排序
expression_df = pd.DataFrame(list(average_expression.items()), columns=['Gene', 'AverageExpression'])

# 按表达量排序，并选择前300个基因
top_300_genes = expression_df.sort_values(by='AverageExpression', ascending=False).head(300)
top_300_gene_list = top_300_genes['Gene'].tolist()

# Step 4: 针对每个病人提取top_300_gene_list对应的基因表达值，并保存到.h5ad文件
for i in file_path:
    if i == 'NC_BRCA':
        continue

    gene_path = path + i
    print(f"Processing and saving gene expressions for slice: {i}")

    # 读取 Visium 数据
    adata = sc.read_visium(path=gene_path, count_file='filtered_feature_bc_matrix.h5', load_images=True)

    # 确保基因名称唯一
    adata.var_names_make_unique()

    # 替换指定病人的基因名
    if i in ['GSM5420753', 'BRCA_BlockASection1_10x', 'BRCA_BlockASection2_10x',
             'BRCA_DuctalCarcinomaInSitu_InvasiveCarcinoma_10x_FFPE', 'BRCA_WholeTranscriptomeAnalysis_10x',
             'BRCA_InvasiveDuctalCarcinoma_StainedWithFluorescent_CD3Antibody_10x',
             'BRCA_Visium_FreshFrozen_WholeTranscriptome_10x']:
        adata.var_names = [replacement_dict.get(gene, gene) for gene in adata.var_names]

    # 再次确保基因名称唯一
    adata.var_names_make_unique()

    # 提取前300个基因的表达矩阵
    top_300_expression_matrix = adata[:, top_300_gene_list].X.toarray()



    # 创建 AnnData 对象并保存为 .h5ad 文件
    # patient_adata = sc.AnnData(X=top_300_expression_matrix, var=pd.DataFrame(index=top_300_gene_list))
    adata = adata[:, top_300_gene_list]  # 只保留前300个基因的数据
    adata.X = top_300_expression_matrix  # 更新表达矩阵
    adata.var.index = top_300_gene_list  # 更新基因名

    # # 保存每个病人的 .h5ad 文件
    output_file = f'./data/Breast_cancer/{i}/filtered_feature_bc_matrix_decoder.h5ad'
    adata.write(output_file)
    print(f"Expression data for {i} saved to {output_file}")
