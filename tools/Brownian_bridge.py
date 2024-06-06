# coding=utf-8
import torch
import random

# Process-wised Regularization
class PRT(torch.nn.Module):
    def __init__(self, cfg):
        self.cfg = cfg
        self.beta = cfg.TRAINING.beta

    def Brownian_bridge_distance(self, emb, bridge):
        bh,bp,bt = bridge[0],bridge[1],bridge[2]
        alpha = torch.true_divide(bp-bh,bt-bh) 
        sigma = alpha * (bt-bp)
        x = emb[1] - (1-alpha)*emb[0] - alpha*emb[2]
        return -torch.norm(x,p=2)**2 / (2*sigma**2)


    def random_select_negative(self, matrix, pos_inx):
        valid_positions = [] 
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if i!=pos_inx[0] and j!=pos_inx[1]:
                    valid_positions.append((i, j))
        random_position = random.choice(valid_positions)
        selected_value = matrix[random_position[0]][random_position[1]]
        return selected_value


    def forward(self, bridges, b_inx):
        batch_size, num_seg, _ = bridges.shape
        assert num_seg > 2
        loss = 0 
        for i in range(batch_size):
            cur_bridge_head = bridges[i][0]
            cur_bridge_head_tiemstamp = b_inx[i][0]
            cur_bridge_tail = bridges[i][-1]
            cur_bridge_tail_tiemstamp = b_inx[i][-1]

            for j in range(1,num_seg-1):
                pos_inx = [i,j]
                
                cur_target_tiemstamp = b_inx[i][j]
                bridge_inx = [cur_bridge_head_tiemstamp, cur_target_tiemstamp, cur_bridge_tail_tiemstamp]

                cur_positive = bridges[i][j]
                positive_emb = [cur_bridge_head, cur_positive, cur_bridge_tail]

                cur_negative = self.random_select_negative(bridges,pos_inx)
                negative_emb = [cur_bridge_head, cur_negative, cur_bridge_tail]

                pos_dis = self.Brownian_bridge_distance(positive_emb, bridge_inx)
                neg_dis = self.Brownian_bridge_distance(negative_emb, bridge_inx)

                cur_loss = pos_dis - neg_dis + self.beta
                if cur_loss > 0:
                    loss += cur_loss
        return loss / batch_size

