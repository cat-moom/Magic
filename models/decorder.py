import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerDecoderModel(nn.Module):
    def __init__(self, input_dim=512, intermediate_dims=[1024, 1500], output_dim=300, nhead=5, num_layers=6,
                 dim_feedforward=2048, dropout=0.1):
        super(TransformerDecoderModel, self).__init__()

        # 定义多层线性层，逐步将输入从512维映射到2000维
        layers = []
        current_dim = input_dim
        for dim in intermediate_dims:
            layers.append(nn.Linear(current_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.LeakyReLU())
            # layers.append(nn.ReLU())  # 增加非线性激活层
            current_dim = dim

        # 最后将线性层映射到目标300维
        layers.append(nn.Linear(current_dim, output_dim))
        layers.append(nn.BatchNorm1d(output_dim))
        layers.append(nn.LeakyReLU())
        self.linear_layers = nn.Sequential(*layers)

        # Transformer解码器层
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=output_dim,  # 输入的维度
            nhead=nhead,  # 注意力头的数量
            dim_feedforward=dim_feedforward,  # 前馈网络的维度
            dropout=dropout  # Dropout概率
        )

        # 堆叠多个解码器层
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # 定义输出层
        self.linear_out = nn.Linear(output_dim, output_dim)

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """
        使用Xavier初始化线性层的权重，常规初始化偏置。
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)  # Xavier初始化
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)  # 偏置初始化为0
            elif isinstance(m, nn.TransformerDecoderLayer):
                for submodule in m.modules():
                    if isinstance(submodule, nn.Linear):
                        nn.init.xavier_uniform_(submodule.weight)
                        if submodule.bias is not None:
                            nn.init.constant_(submodule.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, encoded_features, original_data):
        """
        :param encoded_features: 编码器输出的低维特征, shape (batch_size, 512)
        :param original_data: 原始高维特征, shape (batch_size, 2000)
        """
        original_data = original_data.float()

        # 通过多层线性层逐步映射到高维特征空间
        tgt = self.linear_layers(encoded_features)  # (batch_size, 2000)

        # Transformer解码器需要目标序列（tgt）和记忆序列（memory）
        # 此处为了简化，假设memory等同于tgt
        memory = tgt.unsqueeze(0)  # 增加时间维度 (1, batch_size, 2000)
        tgt = tgt.unsqueeze(0)  # (1, batch_size, 2000)

        # 通过Transformer解码器解码
        decoded = self.transformer_decoder(tgt, memory).squeeze(0)  # (batch_size, 2000)

        # 输出层
        output = self.linear_out(decoded)  # (batch_size, 2000)

        # 计算损失（采用MSE损失）
        loss = F.mse_loss(output, original_data)

        return output, loss


if __name__ == '__main__':
    # 测试模型
    batch_size = 32
    input_dim = 512
    output_dim = 300

    # 模拟输入数据
    encoded_features = torch.randn(batch_size, input_dim)  # 编码后的低维特征
    original_data = torch.randn(batch_size, output_dim)  # 原始的高维特征

    # 创建模型
    model = TransformerDecoderModel(input_dim=input_dim, output_dim=output_dim)

    # 前向传播
    decoded_output, loss = model(encoded_features, original_data)

    print(f"Decoded Output Shape: {decoded_output.shape}")
    print(f"Loss: {loss.item()}")


