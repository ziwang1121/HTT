from __future__ import division, absolute_import
import torch
import torch.nn as nn


class MMTripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    
    Reference:
        Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    
    Imported from `<https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py>`_.
    
    Args:
        margin (float, optional): margin for triplet. Default is 0.3.
    """

    def __init__(self, margin=1):
        super(MMTripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=1)

    def forward(self, inputs_R, inputs_N, inputs_T, targets):
        """
        Args:
            inputs (torch.Tensor): feature matrix with shape (batch_size, feat_dim).
            targets (torch.LongTensor): ground truth labels with shape (num_classes).
        """
        n = inputs_R.size(0)

        # Compute pairwise distance, replace by the official when merged
        dist_R = torch.pow(inputs_R, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist_R = dist_R + dist_R.t()
        dist_R.addmm_(inputs_R, inputs_R.t(), beta=1, alpha=-2)
        dist_R = dist_R.clamp(min=1e-12).sqrt() # for numerical stability

        dist_N = torch.pow(inputs_N, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist_N = dist_N + dist_N.t()
        dist_N.addmm_(inputs_N, inputs_N.t(), beta=1, alpha=-2)
        dist_N = dist_N.clamp(min=1e-12).sqrt() # for numerical stability

        dist_T = torch.pow(inputs_T, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist_T = dist_T + dist_T.t()
        dist_T.addmm_(inputs_T, inputs_T.t(), beta=1, alpha=-2)
        dist_T = dist_T.clamp(min=1e-12).sqrt() # for numerical stability

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(min(dist_R[i][mask[i]].max().unsqueeze(0), dist_N[i][mask[i]].max().unsqueeze(0), dist_T[i][mask[i]].max().unsqueeze(0)))
            dist_an.append(max(dist_R[i][mask[i] == 0].min().unsqueeze(0), dist_N[i][mask[i] == 0].min().unsqueeze(0), dist_T[i][mask[i] == 0].min().unsqueeze(0)))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        return self.ranking_loss(dist_an, dist_ap, y)
