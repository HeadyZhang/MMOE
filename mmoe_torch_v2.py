import torch
import torch.nn as nn

class Expert(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(Expert, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.bn2 = nn.BatchNorm1d(output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.bn1(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.bn2(out)
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
    def __init__(self, x_dim, sparse_x_len, emb_dim, num_experts,
                 experts_out, experts_hidden, towers_hidden,
                 tasks=2, use_bn=True):
        super(MMOE, self).__init__()
        self.input_size = sparse_x_len * emb_dim + 1
        self.num_experts = num_experts
        self.experts_out = experts_out
        self.experts_hidden = experts_hidden
        self.tasks = tasks
        self.emb_dim = emb_dim

        self.tower_input_size = self.experts_out
        self.towers_hidden = towers_hidden

        self.emb_resp = nn.Embedding(x_dim, emb_dim)

        self.softmax = nn.Softmax(dim=-1)
        self.Q_w = nn.Linear(in_features=emb_dim, out_features=emb_dim, bias=True)
        self.K_w = nn.Linear(in_features=emb_dim, out_features=emb_dim, bias=True)
        self.V_w = nn.Linear(in_features=emb_dim, out_features=emb_dim, bias=True)

        self.experts = nn.ModuleList(
                [Expert(self.input_size, self.experts_out, self.experts_hidden) for i in range(self.num_experts)])
        self.w_gates = nn.ParameterList(
            [nn.Parameter(torch.randn(self.input_size, num_experts), requires_grad=True) for i in range(self.tasks)])
        self.towers = nn.ModuleList([Tower(self.tower_input_size, 1, self.towers_hidden) for i in range(self.tasks)])

    def self_attn(self, q, k, v):
        Q, K, V = self.Q_w(q), self.K_w(k), self.V_w(v)
        attn_weights = Q.matmul(torch.transpose(K, 1, 2)) / (K.shape[-1] ** 0.5)
        attn_weights = self.softmax(torch.sigmoid(attn_weights))
        outputs = attn_weights.matmul(V)
        return outputs, attn_weights

    def forward(self, x, r):
        x_input = x.int()
        r = r.float()
        x_rep = self.emb_resp(x_input)
        dims = x_rep.size()  ## batch_size * sparse_x_len * emb_size
        # _x_rep = x_rep / torch.linalg.norm(x_rep, dim=1, keepdim=True)
        # outputs, attn_weights = self.self_attn(_x_rep, _x_rep, _x_rep)

        _x_rep = torch.reshape(x_rep, (-1, dims[1] * dims[2]))  ## (batch_size, sparse_x_len * emb_size)
        dense_x = torch.concat([_x_rep, r.reshape(-1, 1)], 1)  ## (batch_size, sparse_x_len * emb_size)

        experts_o = [e(dense_x) for e in self.experts]
        experts_o_tensor = torch.stack(experts_o)

        gates_o = [self.softmax(dense_x @ g) for g in self.w_gates]

        tower_input = [g.t().unsqueeze(2).expand(-1, -1, self.experts_out) * experts_o_tensor for g in gates_o]
        tower_input = [torch.sum(ti, dim=0) for ti in tower_input]

        final_output = [t(ti) for t, ti in zip(self.towers, tower_input)]
        return torch.stack(final_output).squeeze(-1).T

    def cal_loss(self, is_eff_rev_list, is_eff_pic_list, is_rev_sample_weights, is_pic_sample_weights, outputs):
        criterion1 = nn.BCELoss(reduction='mean', weight=is_rev_sample_weights)
        criterion2 = nn.BCELoss(reduction='mean', weight=is_pic_sample_weights)

        loss1 = criterion1((outputs.T)[0], is_eff_rev_list)
        loss2 = criterion2((outputs.T)[1], is_eff_pic_list)
        loss = loss1 * 0.45 + loss2 * 0.55
        return loss 
    

    def cal_loss_v2(self, is_eff_rev_list, is_eff_pic_list, is_high_quality_list, is_gpage_impr_list, 
                    is_rev_sample_weights, is_pic_sample_weights, is_hq_sample_weights, is_gpage_impr_weghts, 
                    outputs
                    ):
        criterion1 = nn.BCELoss(reduction='mean', weight=is_rev_sample_weights)
        criterion2 = nn.BCELoss(reduction='mean', weight=is_pic_sample_weights)
        criterion3 = nn.BCELoss(reduction='mean', weight=is_hq_sample_weights)
        criterion4 = nn.BCELoss(reduction='mean', weight=is_gpage_impr_weghts)

        loss1 = criterion1((outputs.T)[0], is_eff_rev_list)
        loss2 = criterion2((outputs.T)[1], is_eff_pic_list)
        loss3 = criterion3((outputs.T)[2], is_high_quality_list)
        loss4 = criterion4((outputs.T)[3], is_gpage_impr_list)
        loss = loss1 * 0.2 + loss2 * 0.35 + loss3 * 0.25  +  loss4 * 0.2
        return loss 
        
   