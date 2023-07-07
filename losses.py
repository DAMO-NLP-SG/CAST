import torch
import torch.nn as nn
import torch.nn.functional as F


class ATLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.threshold = nn.Threshold(0, 0, inplace=False)

    def forward(self, logits, labels):
        # TH label
        
        #fp_logit = logits[:, 97].unsqueeze(1) 
        #fn_logit = logits[:, 98].unsqueeze(1) 
        #logits[:,1:] = logits[:,1:] + fp_logit - fn_logit
        #print(logits.size())
        #print(labels.size())
        th_label = torch.zeros_like(labels, dtype=torch.float).to(labels)
        th_label[:, 0] = 1.0
        labels[:, 0] = 0.0

        p_mask = labels + th_label
        n_mask = 1 - labels
        n_ex, num_class = logits.size()
        # Rank positive classes to TH
        logit1 = logits - (1 - p_mask) * 1e30
        loss1 = -(F.log_softmax(logit1, dim=-1) * labels).sum(1)

        # Rank TH to negative classes
        logit2 = logits - (1 - n_mask) * 1e30
        loss2 = -(F.log_softmax(logit2, dim=-1) * th_label).sum(1)
        #loss2 = -(F.log_softmax(logit2, dim=-1)[:,0])
        

        # Sum two parts
        loss = loss1 + loss2 
        loss = loss.mean()
        return loss, loss1, loss2

    def get_label(self, logits, num_labels=-1, inference_th=1.0):
        
        #fp_logit = logits[:, 97].clone().unsqueeze(1).clamp(min=0, ) 
        #fn_logit = logits[:, 98].clone().unsqueeze(1).clamp(min=0, )  
        
        #logits[:,1:] = logits[:,1:] - 0.1 * fp_logit 
        #logits = logits[:, :97]
        if type(inference_th) != float:
            inference_th = inference_th.to(logits)
        th_logit = logits[:, 0].unsqueeze(1) * inference_th
        '''
        unpopular_logits_mask = torch.tensor([0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0.,
        0., 1., 0., 0., 1., 1., 1., 0., 0., 1., 0., 1., 1., 0., 1., 1., 0., 1.,
        0., 1., 0., 1., 1., 1., 1., 1., 0., 0., 1., 1., 1., 1., 0., 1., 0., 1.,
        0., 1., 0., 1., 0., 1., 0., 0., 0., 0., 1., 1., 1., 0., 1., 0., 0., 1.,
        1., 1., 0., 1., 0., 1., 0., 0., 0., 1., 1., 0., 1., 1., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0.])
        margin_logits = 0.1 * th_logit * unpopular_logits_mask.to(logits).unsqueeze(0)
        '''
        #logits = logits + margin_logits
        output = torch.zeros_like(logits).to(logits)
        mask = (logits > th_logit)
        if num_labels > 0:
            top_v, _ = torch.topk(logits, num_labels, dim=1)
            top_v = top_v[:, -1]
            mask = (logits >= top_v.unsqueeze(1)) & mask
        output[mask] = 1.0
        output[:, 0] = (output.sum(1) == 0.).to(logits)
        return output

    
class NCRLoss(nn.Module):
    def __init__(self, shift=0.0, isReg=True, eps=1e-8, reduction='mean'):
        super().__init__()
        self.shift = shift
        self.isReg = isReg
        self.eps = eps
        self.reduction = reduction

    def compute_CE(self, x, y):
        """
        Adapted from "Asymmetric Loss For Multi-Label Classification"
        https://arxiv.org/abs/2009.14119
        """
        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Margin Shifting
        if self.shift is not None and self.shift > 0:
            xs_neg = (xs_neg + self.shift).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        return -loss.sum()
    
    def compute_MAE(self, x, y):

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Margin Shifting
        if self.shift is not None and self.shift > 0:
            xs_neg = (xs_neg + self.shift).clamp(max=1)

        # Basic CE calculation
        los_pos = torch.abs( y - xs_pos.clamp(min=self.eps)).mean(1)
        #los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos 

        return -loss.sum()
    
    
    
    def forward(self, logits, labels):
        # Logit margin for pre-defined relations 
        rel_margin = logits[:,1:] - logits[:,0].unsqueeze(1)
        loss = self.compute_MAE(rel_margin.float(), labels[:,1:].float())
        
        if self.isReg: # Enable margin regularization
            # Logit margin for the none class label
            na_margin = logits[:,0] - logits[:,1:].mean(-1) 
            loss += self.compute_CE(na_margin.float(), labels[:,0].float())
        
        if self.reduction == "mean":
            loss /= labels.shape[0]

        return (loss,loss,loss)

    def get_label(self, logits, num_labels=-1, inference_th=1.0):
        """Copied from https://github.com/wzhouad/ATLOP/blob/main/losses.py#L32"""
        th_logit = logits[:, 0].unsqueeze(1) * inference_th
        output = torch.zeros_like(logits).to(logits)
        mask = (logits > th_logit)
        if num_labels > 0:
            top_v, _ = torch.topk(logits, num_labels, dim=1)
            top_v = top_v[:, -1]
            mask = (logits >= top_v.unsqueeze(1)) & mask
        output[mask] = 1.0
        output[:, 0] = (output.sum(1) == 0.).to(logits)
        return output