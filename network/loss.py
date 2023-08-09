import torch.nn as nn
import torch.nn.functional as F
import torch
from mmengine import Registry
import torch.distributed as dist

LOSS_REGISTRY = Registry("LOSS")

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

@LOSS_REGISTRY.register_module()
class consistency_loss(nn.Module):
    def __init__(self,weight,eta=1e-2):
        super().__init__()
        self.weight=weight
        self.eta=eta

    def forward(self,coord,nocs,R,t,s):
        response=torch.bmm(R.transpose(1,2)/s,(coord-t.unsqueeze(dim=1)).transpose(1,2)).transpose(1,2)
        loss=nn.functional.smooth_l1_loss(nocs,response,beta=0.5,reduction='none').flatten(1).mean(-1) #B
        mask=loss>(self.eta)
        valid=mask.float().sum()
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(valid)
            world_size=dist.get_world_size()
        valid=(valid/world_size).clamp(min=1)
        loss[~mask]=0
        loss=(loss.sum())/valid
        return self.weight*(loss)

@LOSS_REGISTRY.register_module()
class consistency_lossv2(consistency_loss):
    def __init__(self,weight=1.0,beta=0.1,loss='smooth') -> None:
        super().__init__(weight)
        self.beta=beta
        if loss=='smooth':
            self.loss_f=torch.nn.SmoothL1Loss(beta=beta)
        else:
            self.loss_f=torch.nn.MSELoss()
    def forward(self,coord,nocs):
        return self.weight*(self.loss_f(coord,nocs))



@LOSS_REGISTRY.register_module()
class chamfer_lossv2(nn.Module):
    def __init__(self,weight,threshold=1.2) -> None:
        super().__init__()
        self.threshold=threshold
        self.weight=weight
        
    def forward(self,coord,gt):
        gt=gt.transpose(2,1)
        coord=coord.transpose(2,1)
        dis=torch.pow(gt.unsqueeze(dim=-1)-coord.unsqueeze(dim=-2),2).sum(dim=1)
        match_gt=torch.amin(dis,dim=-1)
        match_coord=torch.amin(dis,dim=-2)
        res=(match_coord.mean()+match_gt.mean())

        return self.weight*(res)


@LOSS_REGISTRY.register_module()
class r_lossv2(nn.Module):
    def __init__(self,weight=1.0,beta=0.001,loss='smooth') -> None:
        super().__init__()
        self.weight=weight
        self.beta=beta
        self.loss_f=loss
    def forward(self,pred_r,gt_red,gt_green,sym):
        pred_green=pred_r[:,:,1:2] #B,3,1
        pred_red=pred_r[:,:,0:1] #B,3,1
        if self.loss_f=='smooth':
            green_loss=nn.functional.smooth_l1_loss(gt_green,pred_green,beta=self.beta)
        else:
            green_loss=nn.functional.mse_loss(gt_green,pred_green)

        mask=(sym[:,0]==1) #B
        B=mask.shape[0]
        valid=B-(mask).float().sum()
        b=valid.item()==0
        world_size=1
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(valid)
            world_size=dist.get_world_size()
        valid=valid/world_size
        if b:
            red_loss=0
        else:
            if self.loss_f=='smooth':
                red_loss=nn.functional.smooth_l1_loss(gt_red,pred_red,reduction='none',beta=self.beta)[:,:,0].mean(-1) #B
            else:
                red_loss=nn.functional.mse_loss(gt_red,pred_red,reduction='none')[:,:,0].mean(-1) #B
            red_loss[mask]=0
            red_loss=red_loss.sum()/(valid)
        return self.weight*(green_loss+red_loss)


@LOSS_REGISTRY.register_module()
class t_loss(nn.Module):
    def __init__(self,weight=1.0,beta=0.005,loss='smooth') -> None:
        super().__init__()
        self.weight=weight
        self.beta=beta
        if loss=='smooth':
            self.loss_f=torch.nn.SmoothL1Loss(beta=beta)
        else:
            self.loss_f=torch.nn.MSELoss()
    def forward(self,pred_t,t):
        return self.weight*self.loss_f(pred_t,t)

@LOSS_REGISTRY.register_module()
class s_loss(nn.Module):
    def __init__(self,weight=1.0,beta=0.005,loss='smooth') -> None:
        super().__init__()
        self.weight=weight
        self.beta=beta
        if loss=='smooth':
            self.loss_f=torch.nn.SmoothL1Loss(beta=beta)
        else:
            self.loss_f=torch.nn.MSELoss()
    def forward(self,pred_s,s):
        return self.weight*self.loss_f(pred_s,s)
