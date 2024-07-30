import numpy as np
from torch import nn, tensor
import torch
from torch.autograd import Variable


class IDMarginLossNew(nn.Module):
    def __init__(self, margin=1, dist_type='l1'):
        super(IDMarginLossNew, self).__init__()
        self.dist_type = dist_type
        self.margin = margin
        if dist_type == 'l2':
            self.dist = nn.MSELoss(reduction='sum')
        if dist_type == 'cos':
            self.dist = nn.CosineSimilarity(dim=0)
        if dist_type == 'l1':
            self.dist = nn.L1Loss()

    def forward(self, feat1, feat2, feat3, label1):
        # print(label1)
        label_num = len(label1.unique())
        # label_num = len(label1)
        sample_num = len(label1) // label_num
        feat1 = feat1.chunk(label_num, 0)
        feat2 = feat2.chunk(label_num, 0)
        feat3 = feat3.chunk(label_num, 0)

        distance = []
        distance_fianl = []
        for i in range(label_num-1):
            for j in range(i+1, label_num):
                distance = []
                for m in range(sample_num):
                    for n in range(sample_num):
                        distance.append(abs(self.margin - self.dist(feat1[i][m], feat1[j][n])))
                        distance.append(abs(self.margin - self.dist(feat1[i][m], feat2[j][n])))
                        distance.append(abs(self.margin - self.dist(feat1[i][m], feat3[j][n])))
                        distance.append(abs(self.margin - self.dist(feat2[i][m], feat1[j][n])))
                        distance.append(abs(self.margin - self.dist(feat2[i][m], feat2[j][n])))
                        distance.append(abs(self.margin - self.dist(feat2[i][m], feat3[j][n])))
                        distance.append(abs(self.margin - self.dist(feat3[i][m], feat1[j][n])))
                        distance.append(abs(self.margin - self.dist(feat3[i][m], feat2[j][n])))
                        distance.append(abs(self.margin - self.dist(feat3[i][m], feat3[j][n])))
                distance_fianl.append(max(distance))

        dist = sum(distance_fianl)/len(distance_fianl)

        return dist


