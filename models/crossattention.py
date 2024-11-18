import torch
import torch.nn as nn
import math
from math import sqrt

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class SelfAttention(nn.Module):
    def __init__(self, num_attention_heads, input_size, hidden_size, hidden_dropout_prob):
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = hidden_size

        self.query = nn.Linear(input_size, self.all_head_size)
        self.key = nn.Linear(input_size, self.all_head_size)
        self.value = nn.Linear(input_size, self.all_head_size)

        self.attn_dropout = nn.Dropout()

        # 做完self-attention 做一个前馈全连接 LayerNorm 输出
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def _init_weights(self):
        # Initialize query, key, and value layers
        nn.init.xavier_uniform_(self.query.weight)
        nn.init.xavier_uniform_(self.key.weight)
        nn.init.xavier_uniform_(self.value.weight)

        nn.init.constant_(self.query.bias, 0)
        nn.init.constant_(self.key.bias, 0)
        nn.init.constant_(self.value.bias, 0)

        # Initialize dense layer
        nn.init.xavier_uniform_(self.dense.weight)
        nn.init.constant_(self.dense.bias, 0)

    def transpose_for_scores(self, x):
        new_x_shape = (x.size(0), 1, self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor):

        if input_tensor.dim() == 2:
            input_tensor = input_tensor.unsqueeze(1)

        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states.squeeze(1)


class CalculateAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V):
        # Calculate raw attention scores
        attention = torch.matmul(Q, torch.transpose(K, -1, -2))
        # Apply softmax to normalize
        attention = torch.softmax(attention / sqrt(Q.size(-1)), dim=-1)
        # Multiply attention weights by V
        attention = torch.matmul(attention, V)
        return attention


class Multi_CrossAttention(nn.Module):
    """
    forward时，第一个参数用于计算query，第二个参数用于计算key和value
    """

    def __init__(self, hidden_size, all_head_size, head_num):
        super().__init__()
        self.hidden_size = hidden_size  # 输入维度
        self.all_head_size = all_head_size  # 输出维度
        self.num_heads = head_num  # 注意头的数量
        self.h_size = all_head_size // head_num

        assert all_head_size % head_num == 0

        # W_Q,W_K,W_V (hidden_size,all_head_size)
        self.linear_q = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_k = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_v = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_output = nn.Linear(all_head_size, hidden_size)

        # normalization
        self.norm = sqrt(all_head_size)

        self.layer_norm = nn.LayerNorm(hidden_size)

    def _init_weights(self):
        # Initialize weights using Xavier initialization for linear layers
        nn.init.xavier_uniform_(self.linear_q.weight)
        nn.init.xavier_uniform_(self.linear_k.weight)
        nn.init.xavier_uniform_(self.linear_v.weight)
        nn.init.xavier_uniform_(self.linear_output.weight)

    def forward(self, x, y):
        """
        cross-attention: x,y是两个模型的隐藏层，将x作为q的输入，y作为k和v的输入
        """
        batch_size = x.size(0)

        # q_s: [batch_size, num_heads, seq_length, h_size]
        q_s = self.linear_q(x).view(batch_size, -1, self.num_heads, self.h_size).transpose(1, 2)

        # k_s: [batch_size, num_heads, seq_length, h_size]
        k_s = self.linear_k(y).view(batch_size, -1, self.num_heads, self.h_size).transpose(1, 2)

        # v_s: [batch_size, num_heads, seq_length, h_size]
        v_s = self.linear_v(y).view(batch_size, -1, self.num_heads, self.h_size).transpose(1, 2)

        # Perform attention computation
        attention = CalculateAttention()(q_s, k_s, v_s)

        # attention : [batch_size , seq_length , num_heads * h_size]
        attention = attention.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.h_size)

        # output : [batch_size , seq_length , hidden_size]
        output = self.linear_output(attention)

        output = self.layer_norm(output.squeeze(1) + x)

        return output

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.1):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.norm = nn.LayerNorm(dim)  # 添加LayerNorm以便在加残差之前进行标准化

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        ff_output = self.fc2(self.dropout(self.activation(self.fc1(x))))
        return self.norm(x + ff_output)  # 残差连接 + 标准化


class StackedModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, forward_dim, n_heads, n_stacks):
        super(StackedModel, self).__init__()
        self.n_stacks = n_stacks
        self.first_self_attn = SelfAttention(input_size=input_dim, num_attention_heads=n_heads, hidden_dropout_prob=0.5, hidden_size=hidden_dim)  # First stack with dim 2000
        # self.bi_self_att = SelfAttention(input_size=hidden_dim, num_attention_heads=n_heads, hidden_dropout_prob=0.5, hidden_size=hidden_dim)
        self.cross_att = Multi_CrossAttention(hidden_size=hidden_dim, all_head_size=hidden_dim, head_num=n_heads)
        self.feed_forward = FeedForward(dim=hidden_dim, hidden_dim=forward_dim)

        self.stacks = nn.ModuleList()
        if n_stacks > 1:
            for _ in range(n_stacks-1):
                self.stacks.append(nn.ModuleList([
                    SelfAttention(input_size=hidden_dim, num_attention_heads=n_heads, hidden_dropout_prob=0.5,
                                  hidden_size=hidden_dim),
                    Multi_CrossAttention(hidden_size=hidden_dim, all_head_size=hidden_dim, head_num=n_heads),
                    FeedForward(dim=hidden_dim, hidden_dim=forward_dim)
                ]))

    def forward(self, gene_embedding, image_embedding):
        x1 = gene_embedding
        x2 = image_embedding
        x1 = self.first_self_attn(x1)
        x1 = self.cross_att(x1, x2)
        x1 = self.feed_forward(x1)

        if self.n_stacks > 1:
            for i in range(self.n_stacks-1):
                bi_self_att, cross_att, feed_forward = self.stacks[i]
                x1 = bi_self_att(x1)
                x1 = cross_att(x1, x2)
                x1 = feed_forward(x1)

        return x1

if __name__ == '__main__':
    t1 = torch.rand(32, 512)
    t2 = torch.rand(32, 512)


    # model = SelfAttention(num_attention_heads=8, input_size=512, hidden_size=512, hidden_dropout_prob=0.5)
    # model = Multi_CrossAttention(hidden_size=512, all_head_size=512, head_num=8)
    model = StackedModel(input_dim=512, hidden_dim=512, forward_dim=1024, n_stacks=6, n_heads=8)
    print(model(t1, t2).shape)