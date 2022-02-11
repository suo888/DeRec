import numpy as np, pandas as pd
import tqdm, time, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from copy import deepcopy

from Models.utils.layer import Predictor, Attention
from Models.engine import Engine
from Models.Graph.utils import Aggregator


class FinalNet(nn.Module):
    def __init__(self, Sampler, ModelSettings, TrainSettings):
        super().__init__()

        # init args
        self.num_candidate, self.num_user = Sampler.candidate_num, Sampler.user_num  # self.num_candidate  num_user
        self.user_neigh = Sampler.user_neigh_dict  
        self.user_neigh_r = Sampler.user_neigh_r_dict  

        self.item_neigh = Sampler.item_neigh_dict  
        self.item_neigh_r = Sampler.item_neigh_r_dict  

        self.user_friends = Sampler.user_friend_dict  
        self.item_friends = Sampler.item_friend_dict 
        self.device = TrainSettings['device']
        # self.relation_token_1 = 3
        # self.relation_token = 6
        self.embed_dim = 64
        # self.user_embedding = nn.Embedding(self.num_user + 1, self.embed_dim).to(self.device) 
        # self.item_embedding = nn.Embedding(self.num_candidate + 1, self.embed_dim).to( self.device) 

        self.u2e = nn.Embedding(self.num_user + 1, self.embed_dim).to(self.device) 
        self.v2e = nn.Embedding(self.num_candidate + 1, self.embed_dim).to(self.device)  
        self.r2e = nn.Embedding(7, self.embed_dim).to(self.device)

        self.hid_dim = eval(ModelSettings['hidden_dim'])  # 64
        embed_dim = eval(ModelSettings['embed_dim'])  # 64
        dropout = eval(ModelSettings['dropout'])
        self.num_layer = eval(ModelSettings['num_layer'])
        attn_drop = eval(ModelSettings['attn_drop'])
        aggregator_type = ModelSettings['aggregator_type']
        self.fusion_type = ModelSettings['fusion_type']
        self.f_fusion_type = ModelSettings['f_fusion_type']

        self.activation = nn.LeakyReLU()

        # init embeddings layer

        self.user_soc_GNNs = nn.ModuleList()
        for i in range(self.num_layer):
            self.user_soc_GNNs.append(Aggregator(embed_dim, embed_dim, attn_drop, aggregator_type=aggregator_type))

        self.item_soc_GNNs = nn.ModuleList()
        for i in range(self.num_layer):
            self.item_soc_GNNs.append(Aggregator(embed_dim, embed_dim, attn_drop, aggregator_type=aggregator_type))

        self.user_sim_GNNs = nn.ModuleList()
        for i in range(self.num_layer):
            self.user_sim_GNNs.append(Aggregator(embed_dim, embed_dim, attn_drop, aggregator_type=aggregator_type))

        self.item_sim_GNNs = nn.ModuleList()
        for i in range(self.num_layer):
            self.item_sim_GNNs.append(Aggregator(embed_dim, embed_dim, attn_drop, aggregator_type=aggregator_type))

        self.user_friend_att = Attention(ModelSettings)

        all_layer = self.num_layer + 1  # self.num_layer:2 all_layer：3
        s_dim = 48
        self.all_layer = all_layer
        self.Predictor_1 = Predictor(self.hid_dim, s_dim)  
        self.Predictor_2 = Predictor(self.hid_dim, s_dim)
        self.Predictor_3 = Predictor(self.hid_dim, s_dim)

        # self.init_weights()

        # dui
        self.percent = 0.4

        self.relation_att = torch.randn(self.embed_dim * 2, requires_grad=True).to(self.device)
        self.transform_user = torch.randn(self.embed_dim, self.embed_dim, requires_grad=True).to(self.device)
        self.transform_item = torch.randn(self.embed_dim, self.embed_dim, requires_grad=True).to(self.device)
        self.linear = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.softmax1 = nn.Softmax(dim=0)
        self.softmax2 = nn.Softmax(dim=0)
        self.linear1 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.linear2 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.criterion = nn.MSELoss()

        self.att1 = nn.Linear(self.embed_dim * 2, 1)
        # self.att2 = nn.Linear(self.embed_dim, self.embed_dim)
        # self.att3 = nn.Linear(self.embed_dim, 1)
        self.softmax = nn.Softmax(0)
        self.att5 = nn.Linear(self.embed_dim * 2, 1)
        # self.att6 = nn.Linear(self.embed_dim, 1)

    def graph_aggregate(self, g, GNNs, node_embedding, mode='train', Type=''):
        g = g.local_var()
        init_embed = node_embedding

        # run GNN
        all_embed = [init_embed]
        for l in range(self.num_layer):
            GNN_layer = GNNs[l]
            init_embed = GNN_layer(mode, g, init_embed)
            norm_embed = F.normalize(init_embed, p=2, dim=1)
            all_embed.append(norm_embed)

        all_embed = torch.stack(all_embed, dim=0)
        all_embed = torch.mean(all_embed, dim=0)
        # all_embed = torch.cat(all_embed, dim=1)
        return all_embed


    def get_embeds_u(self, user, candidate, user_global_embedding, item_global_embedding):
        tmp_history_uv = []
        tmp_history_uv_1 = []
        tmp_history_r = []
        tmp_adj = []
        total_social = list(self.user_friends.keys())
        total_inter = list(self.user_neigh.keys())
        for i in range(len(user)):
            if (int(user[i]) in total_inter):
                tmp_history_uv.append(list(self.user_neigh[int(user[i])]))   # tmp_history_uv：保存着每个用户交互的物品集

                tmp_history_r.append(self.user_neigh_r[int(user[i])])  # tmp_history_r：保存着每个用户交互的物品集对应的得分
            else:
                tmp_history_uv.append([0])
                tmp_history_r.append([0])

            if (int(user[i]) in total_social):
                tmp_adj.append(list(self.user_friends[int(user[i])]))  # tmp_adj：保存着每个用户的邻居集合
            else:
                tmp_adj.append([0])

            # self_feats = self.u2e.weight[user]
            # target_feats = self.v2e.weight[candidate]
            # self_feats = torch.LongTensor(user).to(self.device)
            self_feats = user_global_embedding[user]
            #
            # target_feats = torch.LongTensor(candidate).to(self.device)
            target_feats = item_global_embedding[candidate]

        embed_matrix = torch.zeros(len(tmp_history_uv), self.embed_dim, dtype=torch.float).to(self.device)  # 128*64  个0张量
        query = self.linear(torch.cat((self_feats, target_feats),
                                      dim=-1))  # 128*64 user_global：128个用户embedding  item_sim_global:128个物品embedding
        for i in range(len(tmp_history_uv)):  # 循环128次
            history = tmp_history_uv[i]  # 取出第一个列表有70个物品id  history:[25192, 51574, 51595,...
            # for res in self.friends_items[int(i)]:
            #     history.extend(res)
            # history.extend(self.friends_items[int(i)])
            num_histroy_item = len(history)  # 159
            tmp_label = tmp_history_r[i]  # 取出第一个列表有70个物品评分
            # for res in self.friends_items[int(i)]:
            #     tmp_label.extend([self.relation_token_1] * len(res))

            # a = [0,1]
            # e_uv = self.v2e.weight[history]  # e_uv：torch.Size([70, 64]) 第一个列表70个物品的embedding
            # # e_neighbor = user_global_embedding[tmp_adj[i]]  # e_neighbor： torch.Size([88, 256])即第一个用户有社交关系的206个用户的embedding
            # e_neighbor = self.u2e.weight[tmp_adj[i]]

            e_uv = torch.LongTensor(history).to(self.device)
            e_uv = nn.functional.embedding(e_uv, item_global_embedding)  # e_uv：torch.Size([70, 64]) 第一个列表70个物品的embedding


            e_neighbor = torch.LongTensor(tmp_adj[i]).to(self.device)
            e_neighbor = nn.functional.embedding(e_neighbor,
                                                 user_global_embedding)  # e_neighbor：torch.Size([59, 64]) 第一个列表59个邻居的embedding

            # e_uv = torch.cat((e_uv, e_neighbor), 0)  # torch.Size([159, 256])

            # tmp_label += [self.relation_token] * len(tmp_adj[i]) 
            num_histroy_item += len(tmp_adj[i])  # 472
            tmp_label = torch.LongTensor(tmp_label).to(self.device)
            # e_r = self.r2e(tmp_label)
            e_r = nn.functional.embedding(tmp_label, self.r2e.weight)
            if num_histroy_item != 0:
                # agg = self.neighbor_agg(query[i], e_uv, e_r, percent)
                # 先计算交互物品的权重
                prob = -torch.norm(query[i] - e_uv, dim=1)  # 一维 159 e_uv:torch.Size([226, 64])
                # query[i] - e_uv：453*64  dim = 1：每行的数据进行2范数运算，得到453行  所以，prob:一维 453列
                prob = self.softmax1(prob)
                neighbor_selected = torch.multinomial(prob, max(1, int(self.percent * len(e_uv))))  # 一维 63 采样了63个邻居
                relation_selected = e_r[neighbor_selected]  # torch.Size([63, 256])
                neighbor_selected = e_uv[neighbor_selected]  # torch.Size([63, 256])
                selected = torch.cat((neighbor_selected, relation_selected), 1)  # torch.Size([63, 512])
                selected = torch.mm(selected, self.relation_att.unsqueeze(0).t()).squeeze(-1)  # 又降到一维 63列   90*128  128*1
                prob = self.softmax2(selected)

                e_neigh = torch.mm(e_neighbor, self.transform_user)  # e_neighbor:49*64  64*64   49*64
                prob_1 = -torch.norm(query[i] - e_neigh, dim=1)  # 一维 159    这里q-hi时  要转换一下

                prob_1 = self.softmax1(prob_1)
                neighbor_selected_1 = torch.multinomial(prob_1, max(1, int(self.percent * len(e_neighbor))))  # 一维  采样了67个邻居
                # relation_selected = e_r[neighbor_selected]  # torch.Size([67, 64])
                neighbor_selected_1 = e_neighbor[neighbor_selected_1]  # torch.Size([67, 64])
                self_featss = self_feats[i].repeat(len(neighbor_selected_1), 1)
                selected_1 = torch.cat((self_featss, neighbor_selected_1), 1)  # selected:torch.Size([67, 128])
                x = F.relu(self.att1(selected_1))  # cat后放到一个全连接层  维数减一半 5*64
                # x = F.dropout(x, training=self.training)  # 5*64 维度没变 里面的数值变了
                # x = F.relu(self.att2(x))  # 再经过一个全连接层  维数不变 5*64
                # x = F.dropout(x, training=self.training)  # 做两层全连接层
                # x = self.att3(x)  # 输出层 维数直接变为1 5*1
                prob_1 = F.softmax(x.squeeze(1), dim=0)  # x:12*1 att:12*1

                # agg = (torch.mm(neighbor_selected.transpose(0, 1), prob.unsqueeze(-1)).squeeze(-1)) + (torch.mm(neighbor_selected_1.transpose(0, 1), prob_1.unsqueeze(-1)).squeeze(-1)) # agg:64维
                # agg = torch.mm(neighbor_selected.t(), prob).t()
                agg_1 = torch.mm(neighbor_selected.transpose(0, 1), prob.unsqueeze(-1)).squeeze(-1)  # 物品空间的聚合：公式AGG=a*hi   neighbor_selected：采样后的hi 181*64     最后：64*181 交积 181*1  变成一维 64
                agg_2 = torch.mm(neighbor_selected_1.transpose(0, 1), prob_1.unsqueeze(-1)).squeeze(-1)  # 社交空间的聚合：公式AGG=a*hi  一维 64

                agg_1_1 = torch.stack((agg_1, agg_2), 0)  # torch.Size([2, 64])  就是hN*
                self_featss_1 = self_feats[i].repeat(2, 1)   # 复制2行 torch.Size([2, 64])  就是hq
                selected_2 = torch.cat((self_featss_1, agg_1_1), 1)  # selected_2: torch.Size([2, 128])  串联操作
                x = F.relu(self.att5(selected_2))  # cat后放到一个全连接层  维数减一半 5*64
                # x = F.dropout(x, training=self.training)  # 做两层全连接层
                # x = self.att6(x)  # 输出层 维数直接变为1 2*1
                prob_1_1 = F.softmax(x.squeeze(1), dim=0)  # x:12*1 att:12*1    得到注意力b:  prob_1_1:tensor([0.5024, 0.4976]

                agg_b = torch.mm(agg_1_1.transpose(0, 1), prob_1_1.unsqueeze(-1)).squeeze(-1)

                embed_matrix[i] = agg_b
                # agg = torch.cat((agg_1, agg_2), dim=-1)
                # agg = F.relu(self.linear1(agg))
                # embed_matrix[i] = agg
        to_feats = embed_matrix  # torch.Size([128, 256])
        combined = torch.cat((self_feats, to_feats),   dim=-1)  # user_global：torch.Size([128, 256]) user_global就是self_feats  item_sim_global就是target_feats
        # combined = 0.5*(self_feats + to_feats)
        combined = F.relu(self.linear2(combined))  # linear1：将512降到256
        return combined

    def get_embeds_v(self, user, candidate, user_global_embedding, item_global_embedding):
        tmp_history_uv = []
        tmp_history_r = []
        tmp_adj = []
        total_social = list(self.item_friends.keys())  # total_social：所有有社交关系的物品id
        total_inter = list(self.item_neigh.keys())  # total_social：所有有交互行为的物品id
        for i in range(len(candidate)):
            if (int(candidate[i]) in total_inter):
                tmp_history_uv.append(list(self.item_neigh[int(candidate[i])]))  # tmp_history_uv：保存着每个物品交互的用户集
                tmp_history_r.append(self.item_neigh_r[int(candidate[i])])  # tmp_history_r：保存着每个物品交互的物品集对应的得分
            else:
                tmp_history_uv.append([0])
                tmp_history_r.append([0])

            if (int(candidate[i]) in total_social):
                tmp_adj.append(list(self.item_friends[int(candidate[i])]))  # tmp_adj：保存着每个用户的邻居集合
            else:
                tmp_adj.append([0])

            # self_feats = self.v2e.weight[candidate]
            # target_feats = self.u2e.weight[user]
            # self_feats = torch.LongTensor(candidate).to(self.device)
            self_feats = item_global_embedding[candidate]
            #
            # target_feats = torch.LongTensor(user).to(self.device)
            target_feats = user_global_embedding[user]

        embed_matrix = torch.zeros(len(tmp_history_uv), self.embed_dim, dtype=torch.float).to(self.device)  # 128*256  个0张量
        query = self.linear(torch.cat((self_feats, target_feats),
                                      dim=-1))  # 128*256 user_global：128个用户embedding  item_sim_global:128个物品embedding
        for i in range(len(tmp_history_uv)):  # 循环128次
            history = tmp_history_uv[i]  # 取出第一个列表有3个用户id  history:[25192, 51574, 51595,...
            num_histroy_item = len(history)  # 3
            tmp_label = tmp_history_r[i]  # 取出第一个列表有3个物品评分
            # a = [0,1]
            # e_uv = self.u2e.weight[history]  # e_uv：torch.Size([3, 256]) 第一个列表3个用户的embedding  这已经出错了
            # e_neighbor = self.v2e.weight[tmp_adj[i]]  # e_neighbor： torch.Size([39, 256])即第一个物品有社交关系的39个物品的embedding

            e_uv = torch.LongTensor(history).to(self.device)
            e_uv = nn.functional.embedding(e_uv, user_global_embedding)

            e_neighbor = torch.LongTensor(tmp_adj[i]).to(self.device)
            e_neighbor = nn.functional.embedding(e_neighbor, item_global_embedding)

            # e_uv = torch.cat((e_uv, e_neighbor), 0)  # torch.Size([42, 256])

            # tmp_label += [self.relation_token] * len(tmp_adj[i])  # 42个评分 后边的全部是评分6 [4, 5, 5, 5, 4, 4, 4, 5, 4, 4, 5, 5,
            num_histroy_item += len(tmp_adj[i])  # 42
            tmp_label = torch.LongTensor(tmp_label).to(self.device)
            # e_r = self.r2e(tmp_label)
            e_r = nn.functional.embedding(tmp_label, self.r2e.weight)
            if num_histroy_item != 0:
                # agg = self.neighbor_agg(query[i], e_uv, e_r, percent)
                # 先计算交互用户的权重
                prob = -torch.norm(query[i] - e_uv, dim=1)  # 一维 159
                prob = self.softmax1(prob)
                neighbor_selected = torch.multinomial(prob, max(1, int(self.percent * len(e_uv))))  # 一维 63 采样了63个邻居
                relation_selected = e_r[neighbor_selected]  # torch.Size([63, 256])
                neighbor_selected = e_uv[neighbor_selected]  # torch.Size([63, 256])
                selected = torch.cat((neighbor_selected, relation_selected), 1)  # torch.Size([63, 512])
                selected = torch.mm(selected, self.relation_att.unsqueeze(0).t()).squeeze(-1)  # 又降到一维 63列
                prob = self.softmax2(selected)

                # e_neigh = torch.mm(e_neighbor, self.transform.unsqueeze(0).t())
                # prob_1 = -torch.norm(query[i] - e_neigh, dim=1)  # 一维 159
                e_neigh = torch.mm(e_neighbor, self.transform_item)  # e_neighbor:49*64  64*64   49*64
                prob_1 = -torch.norm(query[i] - e_neigh, dim=1)  # 一维 159
                prob_1 = self.softmax1(prob_1)
                neighbor_selected_1 = torch.multinomial(prob_1, max(1, int(self.percent * len(e_neighbor))))  # 一维  采样了67个邻居 报错
                # relation_selected = e_r[neighbor_selected]  # torch.Size([67, 64])
                neighbor_selected_1 = e_neighbor[neighbor_selected_1]  # torch.Size([67, 64])
                self_featss = self_feats[i].repeat(len(neighbor_selected_1), 1)
                selected_1 = torch.cat((self_featss, neighbor_selected_1), 1)  # selected:torch.Size([67, 128])
                x = F.relu(self.att1(selected_1))  # cat后放到一个全连接层  维数减一半 5*64
                # x = F.dropout(x, training=self.training)  # 5*64 维度没变 里面的数值变了
                # x = F.relu(self.att2(x))  # 再经过一个全连接层  维数不变 5*64
                # x = F.dropout(x, training=self.training)  # 做两层全连接层
                # x = self.att3(x)  # 输出层 维数直接变为1 5*1
                prob_1 = F.softmax(x.squeeze(1), dim=0)  # x:12*1 att:12*1

                # agg = (torch.mm(neighbor_selected.transpose(0, 1), prob.unsqueeze(-1)).squeeze(-1)) + (
                #     torch.mm(neighbor_selected_1.transpose(0, 1), prob_1.unsqueeze(-1)).squeeze(-1))  # agg:64维
                # # agg = torch.mm(neighbor_selected.t(), prob).t()
                agg_1 = torch.mm(neighbor_selected.transpose(0, 1), prob.unsqueeze(-1)).squeeze(-1)
                agg_2 = torch.mm(neighbor_selected_1.transpose(0, 1), prob_1.unsqueeze(-1)).squeeze(-1)  # agg:64维

                agg_1_1 = torch.stack((agg_1, agg_2), 0)  # 128维
                self_featss_1 = self_feats[i].repeat(2, 1)
                selected_2 = torch.cat((self_featss_1, agg_1_1), 1)  # selected:torch.Size([2, 128])
                x = F.relu(self.att5(selected_2))  # cat后放到一个全连接层  维数减一半 5*64
                # x = F.dropout(x, training=self.training)  # 做两层全连接层
                # x = self.att6(x)  # 输出层 维数直接变为1 2*1
                prob_1_1 = F.softmax(x.squeeze(1), dim=0)  # x:12*1 att:12*1

                agg_b = torch.mm(agg_1_1.transpose(0, 1), prob_1_1.unsqueeze(-1)).squeeze(-1)

                embed_matrix[i] = agg_b
                # agg = torch.cat((agg_1, agg_2), dim=-1)
                # agg = F.relu(self.linear1(agg))
                # embed_matrix[i] = agg
        to_feats = embed_matrix  # torch.Size([128, 256])
        combined = torch.cat((self_feats, to_feats), dim=-1)  # user_global：torch.Size([128, 256]) user_global就是self_feats  item_sim_global就是target_feats
        # combined = 0.5 * (self_feats + to_feats)
        combined = F.relu(self.linear2(combined))  # linear1：将512降到256  combined：torch.Size([128, 256])
        return combined

    def forward(self, user, candidate, label, user_soc_g, item_soc_g, user_sim_g):
        """
        user: (batch_size)
        candidate: (batch_size, candidate_num)
        """
        user = user.squeeze()
        candidate = candidate.squeeze()
        batch_size = len(user)

        ## embedding layer
        # user_id_embedding = self.user_embedding(user_soc_g.ndata['id'])  # torch.Size([7376, 64])
        user_id_embedding = self.u2e(user_soc_g.ndata['id'])
        # item_id_embedding = self.item_embedding(item_sim_g.ndata['id'])  # torch.Size([106798, 64])
        item_id_embedding = self.v2e(item_soc_g.ndata['id'])

        ### user social
        user_soc_embedding = self.graph_aggregate(user_soc_g, self.user_soc_GNNs, user_id_embedding,
                                                  Type='user_soc')  # torch.Size([7376, 256])
        user_soc_embedding[0] = torch.zeros_like(user_soc_embedding[0])
        # user sim
        user_sim_embedding = self.graph_aggregate(user_sim_g, self.user_sim_GNNs, user_id_embedding,
                                                  Type='user_sim')  # torch.Size([7376, 256])
        user_sim_embedding[0] = torch.zeros_like(user_sim_embedding[0])
        # item social
        item_soc_embedding = self.graph_aggregate(item_soc_g, self.item_soc_GNNs, item_id_embedding,
                                                  Type='item_soc')  # torch.Size([7376, 256])
        item_soc_embedding[0] = torch.zeros_like(item_soc_embedding[0])


        ### Global info
        user_global_embedding = 0.5 * (user_soc_embedding + user_sim_embedding)

        item_global_embedding = item_soc_embedding

        embeds_u = self.get_embeds_u(user, candidate, user_global_embedding, item_global_embedding)  # embeds_u：torch.Size([128, 256]) tensor([[0.0000, 0.0000, 0.0389,  ..., 0.0230, 0.0000, 0.0609],
        embeds_v = self.get_embeds_v(user, candidate, user_global_embedding, item_global_embedding)
        # user_final = 0.5*(user_global+embeds_u)
        # item_final = 0.5*(item_sim_global+embeds_v)
        scores = torch.mm(embeds_u, embeds_v.t()).diagonal()  # scores 128维  tensor([0.9609, 1.5715, 2.5564, 3.7068, 1.6809, 1.5378, 1.3763, 5.7712, 2.0401,
        return scores


class FinalEngine(Engine):

    def __init__(self, Sampler, DataSettings, TrainSettings, ModelSettings, ResultSettings):
        self.Sampler = Sampler
        self.model = FinalNet(Sampler, ModelSettings, TrainSettings)
        self.model.to(TrainSettings['device'])
        self.batch_size = eval(TrainSettings['batch_size'])
        self.eval_neg_num = eval(DataSettings['eval_neg_num'])
        self.eval_ks = eval(TrainSettings['eval_ks'])
        self.user_history_sample_num = eval(DataSettings['user_history_sample_num'])
        self.item_history_sample_num = eval(DataSettings['item_history_sample_num'])
        self.user_friend_sample_num = eval(DataSettings['user_friend_sample_num'])
        super(FinalEngine, self).__init__(TrainSettings, ModelSettings, ResultSettings)

        ### neighbor data
        self.user_neigh = Sampler.user_neigh_dict
        self.item_neigh = Sampler.item_neigh_dict
        self.user_friends = Sampler.user_friend_dict
        self.criterion = nn.MSELoss()

    def history_sample(self, ks, neigh_dict, masks, mode='train', Type='user'):
        k_neighs = [[] if x not in neigh_dict.keys() else deepcopy(neigh_dict[x]) for x in ks]
        if Type == 'user':
            h_sample_num = self.user_history_sample_num
        else:
            h_sample_num = self.item_history_sample_num

        neigh_lens = []
        r_max = 0
        for i in range(len(k_neighs)):
            mask = masks[i]
            # if mask in k_neighs[i]:
            #     k_neighs[i].remove(mask)

            if k_neighs[i] == []:
                k_neighs[i] = [0]


        return k_neighs, neigh_lens

    def friend_sample(self, users, user_social, user_history, is_sample=True):
        f_sample_num = self.user_friend_sample_num
        h_sample_num = self.user_history_sample_num
        batch_size = len(users)
        u_friends = [[] if x not in user_social.keys() else deepcopy(user_social[x]) for x in users]
        friends_items = []
        f_max, r_max = 0, 0
        friends_lens = []
        for i in range(batch_size):
            cur_f_items = []

            cur_friend = u_friends[i]
            if is_sample:
                sample_num = min(f_sample_num, len(cur_friend))
                cur_friend = random.sample(cur_friend, sample_num)
            f_max = max(len(cur_friend), f_max)

            for f in cur_friend:
                if f in user_history.keys():

                    tmp_items = user_history[f]
                    if is_sample:
                        sample_num = min(h_sample_num, len(tmp_items))
                        tmp_items = random.sample(tmp_items, sample_num)
                    r_max = max(len(tmp_items), r_max)
                    cur_f_items.append(tmp_items)

            if cur_f_items == []:
                cur_f_items = [[0]]
            friends_items.append(cur_f_items)
            friends_lens.append(len(cur_friend))



        return friends_items, friends_lens


    def loss(self, user, candidate, label, user_soc_g, item_soc_g, user_sim_g):
        scores = self.model(user, candidate, label, user_soc_g, item_soc_g, user_sim_g)

        return self.criterion(scores, label)

    def train(self, train_loader, graphs, epoch_id, best_rmse, best_mae):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        self.model.train()
        device = self.device
        user_soc_g = graphs['user_soc_g'].to(torch.device(device))
        item_soc_g = graphs['item_soc_g'].to(torch.device(device))
        user_sim_g = graphs['user_sim_g'].to(torch.device(device))
        # item_sim_g = graphs['item_sim_g'].to(torch.device(device))
        running_loss = 0.0
        # # total_loss = 0
        # tmp_train_loss = []
        # t0 = time.time()
        for i, input_list in enumerate(tqdm.tqdm(train_loader, desc="train", smoothing=0, mininterval=1.0)):
            batch_users, batch_items = list(input_list[0].numpy()), list(input_list[1].numpy())
            self.model.item_users, self.model.item_users_lens = self.history_sample(batch_items, self.item_neigh,
                                                                                    batch_users, mode='train',
                                                                                    Type='item')
            self.model.user_items, self.model.user_items_lens = self.history_sample(batch_users, self.user_neigh,
                                                                                    batch_items, mode='train',
                                                                                    Type='user')
            self.model.friends_items, self.model.friends_lens = self.friend_sample(batch_users, self.user_friends,
                                                                                   self.user_neigh)

            # run model
            input_list = [x.to(device) for x in input_list]
            self.optimizer.zero_grad()
            # pred_list= self.model(*input_list[0:], user_soc_g, user_sim_g, item_sim_g, mode='train')
            loss = self.loss(*input_list[0:], user_soc_g,item_soc_g, user_sim_g)
            loss.backward()  # 得到梯度 利用反向传播，得到每个参数更新的梯度
            self.optimizer.step()  # 参数进行优化 根据上层得到的梯度，对每个参数进行优化，卷积层的参数进行调整
            running_loss += loss.item()  # 一个元素张量可以用x.item()得到元素值
            if i % 100 == 0:  # # 169992/128=1328.06  所以i=1300的时候 本次epoch结束
                print('[%d, %5d] loss: %.3f, The best rmse/mae: %.6f / %.6f' % (
                    epoch_id, i, running_loss / 100, best_rmse, best_mae))  # 输出
                running_loss = 0.0
        return 0



    def evaluate(self, test_loader, graphs):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        self.model.eval()
        device = self.device
        user_soc_g = graphs['user_soc_g'].to(torch.device(device))
        item_soc_g = graphs['item_soc_g'].to(torch.device(device))
        user_sim_g = graphs['user_sim_g'].to(torch.device(device))
        # item_sim_g = graphs['item_sim_g'].to(torch.device(device))



        tmp_pred = []
        target = []
        with torch.no_grad():
            for test_u, test_v, tmp_target in test_loader:
                batch_users, batch_items = list(test_u.numpy()), list(test_v.numpy())
                self.model.item_users, self.model.item_users_lens = self.history_sample(batch_items, self.item_neigh,
                                                                                        batch_users, mode='eval',
                                                                                        Type='item')
                self.model.user_items, self.model.user_items_lens = self.history_sample(batch_users, self.user_neigh,
                                                                                        batch_items, mode='eval',
                                                                                        Type='user')
                self.model.friends_items, self.model.friends_lens = self.friend_sample(batch_users, self.user_friends,
                                                                                       self.user_neigh)
                test_u, test_v, tmp_target = test_u.to(device), test_v.to(device), tmp_target.to(device)
                val_output = self.model(test_u, test_v, tmp_target, user_soc_g, item_soc_g, user_sim_g)  # scores = self.model(label, user, candidate, user_soc_g, user_sim_g, item_sim_g, mode='train')
                tmp_pred.append(list(val_output.data.cpu().numpy()))
                target.append(list(tmp_target.data.cpu().numpy()))
        tmp_pred = np.array(sum(tmp_pred, []))
        target = np.array(sum(target, []))
        expected_rmse = sqrt(mean_squared_error(tmp_pred, target))
        mae = mean_absolute_error(tmp_pred, target)
        print("验证成功")
        return expected_rmse, mae

