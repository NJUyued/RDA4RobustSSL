import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from train_utils import ce_loss

def normalize_d(x):
    x_sum = torch.sum(x)
    x = x / x_sum
    return x.detach()

class Get_Scalar:
    def __init__(self, value):
        self.value = value
        
    def get_value(self, iter):
        return self.value
    
    def __call__(self, iter):
        return self.value


def consistency_loss_rda(logits_x_ulb_w_reverse, logits_x_ulb_s_reverse, logits_w, logits_s, distri, distri_reverse, name='ce', T=1.0):
    assert name in ['ce', 'L2']
    logits_w = logits_w.detach()
    logits_x_ulb_w_reverse = logits_x_ulb_w_reverse.detach()
    distri = distri.detach()
    distri_reverse = distri_reverse.detach()
    if name == 'L2':
        assert logits_w.size() == logits_s.size()
        return F.mse_loss(logits_s, logits_w, reduction='mean')
    
    elif name == 'L2_mask':
        pass

    elif name == 'ce':
        pseudo_label = torch.softmax(logits_w, dim=-1)
        pseudo_label_reverse = torch.softmax(logits_x_ulb_w_reverse, dim=-1)
   
        distri_ = torch.ones_like(distri) - distri 
        distri_ = normalize_d(distri_)      
        pseudo_label_reverse_da = normalize_d(pseudo_label_reverse * (torch.mean(distri_,dim=0) / torch.mean(distri_reverse,dim=0)))  
        distri_reverse_ = torch.ones_like(distri_reverse) - distri_reverse 
        distri_reverse_ = normalize_d(distri_reverse_)  
        pseudo_label_da = normalize_d(pseudo_label * (torch.mean(distri_reverse_,dim=0) / torch.mean(distri,dim=0)))
        max_probs, max_idx = torch.max(pseudo_label_da, dim=-1)
     
        loss_cd = ce_loss(logits_s, max_idx, use_hard_labels = True, reduction='none') 
        loss_ca = ce_loss(logits_x_ulb_s_reverse, pseudo_label_reverse_da, use_hard_labels = False, reduction='none')   

        return loss_ca.mean(), loss_cd.mean()

    else:
        assert Exception('Not Implemented consistency_loss')


