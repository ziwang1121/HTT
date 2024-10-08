# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import torch
import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from .triplet_loss import TripletLoss
from .center_loss import CenterLoss
import torch.nn as nn
from .multi_modal_margin_loss_new import multiModalMarginLossNew
from .hard_mine_triplet_loss_multi_modal import MMTripletLoss
from .multi_modal_id_margin_loss import IDMarginLossNew
from .center_loss import CenterLoss

def make_loss(cfg, num_classes):    # modified by gu
    sampler = cfg.DATALOADER.SAMPLER
    feat_dim = 2048
    Hint = nn.MSELoss()
    criterion_m = multiModalMarginLossNew(margin=1)
    criterion_i = IDMarginLossNew(margin=0.5, dist_type='l1')
    criterion_c = CenterLoss()
    criterion_t = MMTripletLoss()
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss
    if 'triplet' in cfg.MODEL.METRIC_LOSS_TYPE:
        if cfg.MODEL.NO_MARGIN:
            triplet = TripletLoss()
            print("using soft triplet loss for training")
        else:
            triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
            print("using triplet loss with margin:{}".format(cfg.SOLVER.MARGIN))
    else:
        print('expected METRIC_LOSS_TYPE should be triplet'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        print("label smooth on, numclasses:", num_classes)

    if sampler == 'softmax':
        def loss_func(score, feat, target):
            return F.cross_entropy(score, target)

    elif cfg.DATALOADER.SAMPLER == 'softmax_triplet':
        def loss_func(score, feat, target, target_cam):
            if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    if isinstance(score, list):
                        ID_LOSS = [xent(scor, target) for scor in score[1:]]
                        ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                        ID_LOSS = 0.5 * ID_LOSS + 0.5 * xent(score[0], target)
                    else:
                        ID_LOSS = xent(score, target)

                    if isinstance(feat, list):
                            TRI_LOSS = [triplet(feats, target)[0] for feats in feat[1:]]
                            TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
                            TRI_LOSS = 0.5 * TRI_LOSS + 0.5 * triplet(feat[0], target)[0]
                    else:
                            TRI_LOSS = triplet(feat, target)[0]

                    return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + \
                               cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
                else:
                    if isinstance(score, list):
                        ID_LOSS = [F.cross_entropy(scor, target) for scor in score]
                        ID_LOSS = ID_LOSS
                    else:
                        ID_LOSS = F.cross_entropy(score, target)
                        # print(ID_LOSS)
                        

                    if isinstance(feat, list):
                            IDM_LOSS = criterion_i(F.normalize(feat[1], p=2, dim=1), F.normalize(feat[2], p=2, dim=1), F.normalize(feat[3], p=2, dim=1), target)
                            MMM_LOSS = criterion_m(F.normalize(feat[1], p=2, dim=1), F.normalize(feat[2], p=2, dim=1), F.normalize(feat[3], p=2, dim=1), target)

                            TRI_LOSS = 0.5 * MMM_LOSS + 0.5 * IDM_LOSS + triplet(feat[0], target)[0]
                            # TRI_LOSS = triplet(feat[0], target)[0]
                            # TRI_LOSS = 0.5 * IDM_LOSS + triplet(feat[0], target)[0]
                            # TRI_LOSS = 0.5 * MMM_LOSS + triplet(feat[0], target)[0]
                    else:
                            TRI_LOSS = triplet(feat, target)[0]
                    # print("ID_LOSS", ID_LOSS)
                    # print("IDM_LOSS", IDM_LOSS)
                    # print("MMM_LOSS", MMM_LOSS)
                    # print("TRI_LOSS", TRI_LOSS)
                    # # raise RuntimeError
                    return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + \
                               cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
            else:
                print('expected METRIC_LOSS_TYPE should be triplet'
                      'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    else:
        print('expected sampler should be softmax, triplet, softmax_triplet or softmax_triplet_center'
              'but got {}'.format(cfg.DATALOADER.SAMPLER))
    return loss_func, center_criterion


def make_loss_ttt(cfg, num_classes):    # modified by gu
    sampler = cfg.DATALOADER.SAMPLER
    feat_dim = 2048
    criterion_m = multiModalMarginLossNew(margin=1)
    criterion_i = IDMarginLossNew(margin=1, dist_type='l1')
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss

    if sampler == 'softmax':
        def loss_func(score, feat, target, target_cam):
            # import pdb
            # pdb.set_trace()
            pseudo_label = score.max(1)[1]

            MMM_LOSS = criterion_m(F.normalize(feat[1], p=2, dim=1), F.normalize(feat[2], p=2, dim=1), F.normalize(feat[3], p=2, dim=1), pseudo_label)
            if pseudo_label[0] != pseudo_label[1]:
                IDM_LOSS = criterion_i(F.normalize(feat[1], p=2, dim=1), F.normalize(feat[2], p=2, dim=1), F.normalize(feat[3], p=2, dim=1), pseudo_label)
                TTT_LOSS = MMM_LOSS + IDM_LOSS
            else:
                TTT_LOSS = MMM_LOSS
            return cfg.MODEL.TRIPLET_LOSS_WEIGHT * TTT_LOSS


    else:
        print('expected sampler should be softmax, triplet, softmax_triplet or softmax_triplet_center'
              'but got {}'.format(cfg.DATALOADER.SAMPLER))
    return loss_func, center_criterion
