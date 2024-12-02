import torch
import torch.nn as nn

from torch.nn import functional as F

from ..common.utils import off_diagonal


class TotalCodingRate(nn.Module):
    def __init__(self, eps=0.01):
        super(TotalCodingRate, self).__init__()
        self.eps = eps
        
    def compute_discrimn_loss(self, W):
        """Discriminative Loss."""
        p, m = W.shape  #[d, B]
        I = torch.eye(p,device=W.device)
        scalar = p / (m * self.eps)
        logdet = torch.logdet(I + scalar * W.matmul(W.T))
        return logdet / 2.
    
    def forward(self,X):
        return - self.compute_discrimn_loss(X.T)


class Similarity_Loss(nn.Module):
    def __init__(self, ):
        super().__init__()
        pass

    def forward(self, z_list, z_avg):
        z_sim = 0
        num_patch = len(z_list)
        z_list = torch.stack(list(z_list), dim=0)
        z_avg = z_list.mean(dim=0)
        
        z_sim = 0
        for i in range(num_patch):
            z_sim += F.cosine_similarity(z_list[i], z_avg, dim=1).mean()
            
        z_sim = z_sim/num_patch
        z_sim_out = z_sim.clone().detach()
                
        return -z_sim, z_sim_out


def cal_TCR(z, criterion, num_patches):
    z_list = z.chunk(num_patches,dim=0)
    loss = 0
    for i in range(num_patches):
        loss += criterion(z_list[i])
    loss = loss/num_patches
    return loss


def chunk_avg(x,n_chunks=2,normalize=False):
    x_list = x.chunk(n_chunks,dim=0)
    x = torch.stack(x_list,dim=0)
    if not normalize:
        return x.mean(0)
    else:
        return F.normalize(x.mean(0),dim=1)


class EMPSLLLoss(nn.Module):
    """
    Loss used in the paper 
    "EMP-SSL: Towards Self-Supervised Learning in One Training Epoch"

    Reference: https://github.com/tsb0601/EMP-SSL/
    """

    def __init__(self, num_patches: int = 100, patch_sim: float = 200, tcr: float = 1.0, eps: float = 0.2):
        super().__init__()
        self.num_patches = num_patches        
        self.patch_sim = patch_sim
        self.tcr = tcr
        self.contractive_loss = Similarity_Loss()
        self.criterion = TotalCodingRate(eps=eps)

    def forward(self, z_proj):
        z_list = z_proj.chunk(self.num_patches, dim=0)
        z_avg = chunk_avg(z_proj, self.num_patches)
        
        #Contractive Loss
        loss_contract, _ = self.contractive_loss(z_list, z_avg)
        loss_TCR = cal_TCR(z_proj, self.criterion, self.num_patches)
        
        loss = self.patch_sim*loss_contract + self.tcr*loss_TCR
        return loss