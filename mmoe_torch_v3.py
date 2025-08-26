import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from torch.nn.functional import normalize



class CrossNet(nn.Module):
    """轻量级特征交叉网络(DCN-V2)"""
    def __init__(self, input_dim, num_layers=2):
        super(CrossNet, self).__init__()
        self.layers = nn.ModuleList([
            nn.Linear(input_dim, input_dim) for _ in range(num_layers)
        ])
        
    def forward(self, x):
        x0 = x
        for layer in self.layers:
            x = x0 * layer(x) + x  # 特征交叉公式
        return x


class Expert(nn.Module):
    """共享专家网络（增加残差连接）"""
    def __init__(self, input_size, output_size, hidden_size):
        super(Expert, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.bn2 = nn.BatchNorm1d(output_size)
        # self.shortcut = nn.Linear(input_size, output_size) if input_size != output_size else None
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # residual = self.shortcut(x) if self.shortcut else x
        out = self.fc1(x)
        out = self.relu(out)
        out = self.bn1(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.bn2(out)
        # return out + residual
        return out
    

class Tower(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(Tower, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.bn2 = nn.BatchNorm1d(output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.bn1(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

class MMOE(nn.Module):
    """双独立门控网络的MMOE模型"""
    def __init__(self, 
                 x_dim, 
                 sparse_x_len, 
                 emb_dim, 
                 num_experts,
                 experts_out,
                 experts_hidden, 
                 towers_hidden,
                 tasks=2, 
                 use_bn=True, 
                 use_cross_x=False,
                 alpha = 0.2):
         super(MMOE, self).__init__()
         ## 特征嵌入层
         self.emb = nn.Embedding(x_dim, emb_dim)
         self.sparse_x_len = sparse_x_len
         self.emb_dim = emb_dim
         self.tasks = tasks
         self.input_size = sparse_x_len * emb_dim + 1
         self.num_experts = num_experts
         self.experts_out = experts_out
         self.experts_hidden = experts_hidden
         self.tasks = tasks
         self.emb_dim = emb_dim
         self.tower_input_size = self.experts_out
         self.towers_hidden = towers_hidden
         self.use_cross_x = use_cross_x
         
         ## 特征交叉层
         self.cross_dim = sparse_x_len * emb_dim 
         self.cross_net = CrossNet(self.cross_dim, num_layers=2)
         
         self.softmax = nn.Softmax(dim=-1)

         self.experts = nn.ModuleList(
                [Expert(self.input_size, self.experts_out, self.experts_hidden) for _ in range(self.num_experts)])
         self.w_gates = nn.ParameterList(
            [nn.Parameter(torch.randn(self.input_size, num_experts), requires_grad=True) for _ in range(self.tasks)])
         self.towers = nn.ModuleList([Tower(self.tower_input_size, 1, self.towers_hidden) for _ in range(self.tasks)])

         # 动态损失权重参数
         self.alpha = alpha  # GradNorm的超参数
         
    def get_last_shared_layer(self):
        """获取最后一个共享层（用于GradNorm计算）"""
        return self.experts[-1].fc2  # 返回最后一个专家网络的输出层
    
    def forward(self, sparse_x, r):
        # 稀疏特征嵌入
        sparse_x = sparse_x.int()
        r = r.float()
        emb_x = self.emb(sparse_x).view(sparse_x.size(0), -1)
        
        # 特征交叉
        if self.use_cross_x:
            cross_x = self.cross_net(emb_x)
            dense_x = torch.concat([cross_x, r.unsqueeze(1)], 1)
        else:
            dense_x =  torch.concat([emb_x, r.unsqueeze(1)], 1)

        # 专家输出
        experts_o = [e(dense_x) for e in self.experts]
        experts_o_tensor = torch.stack(experts_o)

        gates_o = [self.softmax(dense_x @ g) for g in self.w_gates]

        tower_input = [g.t().unsqueeze(2).expand(-1, -1, self.experts_out) * experts_o_tensor for g in gates_o]
        tower_input = [torch.sum(ti, dim=0) for ti in tower_input]

        final_output = [t(ti) for t, ti in zip(self.towers, tower_input)]
        return torch.stack(final_output).squeeze(-1).T   


    def cal_loss(self, targets, outputs, sample_weights, weights, gamma=2.0):
        task_losses = []

        for i in range(self.tasks):
            bce_loss = F.binary_cross_entropy(outputs[:, i], targets[:, i], sample_weights[:, i])
            task_losses.append(bce_loss * weights[i])
        
        task_losses = torch.stack(task_losses)

        total_loss = torch.sum(task_losses)
        return total_loss 
    
    
    def focal_loss(self, targets, outputs, weights, gamma=2.0):
        """
        结合Focal Loss和GradNorm的动态加权多任务损失
        Args:
            targets: 各任务的真实标签 [batch_size, num_tasks]
            outputs: 各任务的预测输出 [batch_size, num_tasks]
            gamma: Focal Loss的超参数
        """
        task_losses = []

        for i in range(self.tasks):
            bce_loss = F.binary_cross_entropy(outputs[:, i], targets[:, i], reduction='none')
            p = torch.exp(-bce_loss)  # p ≈ predicted probability
            focal_loss = ((1 - p) ** gamma * bce_loss).mean()
            task_losses.append(focal_loss * weights[i])
    
        task_losses = torch.stack(task_losses)
        total_loss = torch.sum(task_losses)
        return total_loss      