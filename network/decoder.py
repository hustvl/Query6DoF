import torch
import torch.nn as nn
import torch.nn.functional as F
from .basic import MLP
from mmengine import Registry
import math

DECODER_REGISTRY = Registry("DECODER")


@DECODER_REGISTRY.register_module()
class deep_prior_decoderv2_9(nn.Module):
    def __init__(self,input_dim=256,group=4,cat_num=6,middle_dim=1024,training=False,vis=False) -> None:
        super().__init__()
        self.input_dim=input_dim
        self.group=group
        self.mlp1=MLP(input_dim,middle_dim,group*input_dim*cat_num)
        self.fc1=torch.nn.Linear(group*input_dim,input_dim)
        self.fc2=torch.nn.Linear(group*input_dim,input_dim)
        self.mlp2=MLP(input_dim,middle_dim,input_dim)
        self.mlp3=MLP(input_dim,middle_dim,group*input_dim*cat_num)
        self.mlp4=MLP(input_dim+64,middle_dim,input_dim+64)
        self.mlp5=MLP(input_dim+64,middle_dim,input_dim*2)
        self.mlp6=MLP(input_dim,middle_dim,input_dim)
        self.training=training
        self.chamfer=MLP(input_dim,128,3)
        self.glo_enhance1=global_enhancev2()
        self.glo_enhance2=global_enhancev2()
        self.glo_enhance3=global_enhancev2()
        self.glo_enhance4=global_enhancev2(input_dim+64)
        self.nocs_mlp=MLP(input_dim+64,128,3)
        self.coord_mlp=MLP(3,32,64)
        self.alpha1=torch.nn.parameter.Parameter(data=torch.tensor([1],dtype=torch.float32),requires_grad=True)
        self.alpha2=torch.nn.parameter.Parameter(data=torch.tensor([1],dtype=torch.float32),requires_grad=True)
        self.vis=vis


    def forward(self,prior_feat,inst_feat,index,encoder_input):
        '''
        inst:B,N,D
        prior:B,M,D
        '''
        B,N,D=inst_feat.shape
        M=prior_feat.shape[1]

        prior_feat=self.glo_enhance1(prior_feat)
        inst_feat=self.glo_enhance2(inst_feat)

        conv_para1=self.mlp1(prior_feat) #B,M,6*D*4
        conv_para1=conv_para1.transpose(1,2).contiguous().view(-1,self.group*D,M)
        conv_para1=torch.index_select(conv_para1,0,index) #B,4*D,M
        conv_para1=(conv_para1.view(B,D,-1)) #B,D,4*M

        iam1=torch.bmm(inst_feat,conv_para1) #B,N,4*M
        iam_prob=(iam1.sigmoid_())/(self.alpha1.clamp(min=1e-5))

        aggre_inst=torch.bmm(iam_prob.transpose(1,2),inst_feat) #B,4*M,D
        normalizer=iam_prob.sum(1,keepdim=True).transpose(1,2).clamp(min=1e-6) #B,4*M,1
        if not self.training and self.vis:
            iam1=iam_prob/(normalizer.transpose(1,2))#B,N,4*M
        aggre_inst=aggre_inst/normalizer
        aggre_inst=aggre_inst.view(B,M,-1) #B,M,4*D
        aggre_inst=self.fc1(aggre_inst) #B,M,D

        prior_feat=prior_feat+aggre_inst
        res_prior_feat=self.mlp2(prior_feat)#B,M,D
        prior_feat=res_prior_feat+prior_feat

        prior_feat=self.glo_enhance3(prior_feat)

        res_prior_feat=self.mlp6(prior_feat)#B,M,D
        prior_feat=res_prior_feat+prior_feat

        if self.training or self.vis:
            coord=self.chamfer(prior_feat)

        conv_para2=self.mlp3(inst_feat) #B,N,6*D*4
        conv_para2=conv_para2.transpose(1,2).contiguous().view(-1,self.group*D,N)
        conv_para2=torch.index_select(conv_para2,0,index) #B,4*D,N
        conv_para2=conv_para2.view(B,D,-1) #B,D,4*N

        iam2=torch.bmm(prior_feat,conv_para2) #B,M,4*N
        iam_prob=(iam2.sigmoid_())/(self.alpha2.clamp(min=1e-5)) #B,M,4*N

        aggre_prior=torch.bmm(iam_prob.transpose(1,2),prior_feat) #B,4*N,D
        normalizer=iam_prob.sum(1,keepdim=True).transpose(1,2).clamp(min=1e-6) #B,4*N,1
        if not self.training and self.vis:
            iam2=iam_prob/(normalizer.transpose(1,2))#B,M,4*N
        aggre_prior=aggre_prior/(normalizer)#B,N,4*M
        aggre_prior=aggre_prior.view(B,N,-1) #B,N,4*D
        aggre_prior=self.fc2(aggre_prior) #B,N,D

        
        inst_feat=inst_feat+aggre_prior#B,N,D

        coord_feat=self.coord_mlp(encoder_input) #B,N,64
        inst_feat=torch.cat((inst_feat,coord_feat),dim=-1)

        res_inst_feat=self.mlp4(inst_feat) #B,N,D
        inst_feat=res_inst_feat+inst_feat

        inst_feat=self.glo_enhance4(inst_feat)


        if self.training or self.vis:
            response_coord=self.nocs_mlp(inst_feat)
            response_coord=response_coord.sigmoid_()-0.5


        inst_feat=self.mlp5(inst_feat)

        inst_feat=torch.nn.functional.adaptive_avg_pool1d(inst_feat.transpose(1,2),1).transpose(1,2) #B,1,D
        if self.training:
            return inst_feat,coord,response_coord
        elif not self.vis:
            return inst_feat
        else:
            return inst_feat,coord,response_coord,iam1,iam2






class global_enhancev2(nn.Module):
    def __init__(self,input_dim=256) -> None:
        super().__init__()
        alpha=torch.tensor([1],dtype=torch.float32)
        self.alpha=torch.nn.parameter.Parameter(data=alpha,requires_grad=True)
        beta=torch.tensor([0],dtype=torch.float32)
        self.beta=torch.nn.parameter.Parameter(data=beta,requires_grad=True)
        self.linear=torch.nn.Conv1d(input_dim,input_dim,1,bias=False)

    def forward(self,feat):
        '''
        feat:B,N,D
        '''
        global_feat=torch.nn.functional.adaptive_avg_pool1d(feat.transpose(1,2),1) #B,D,1
        global_feat=self.linear(global_feat)
        atten=torch.bmm(feat,global_feat) #B,N,1
        mean=atten.squeeze(-1).mean(-1) #B
        std=torch.std(atten.squeeze(-1), dim=-1, unbiased=False) 
        atten=self.alpha*(atten-mean.view(-1,1,1))/(std.view(-1,1,1)+1e-5)+self.beta
        global_feat=torch.bmm(atten,global_feat.transpose(1,2)) #B,N,D
        feat=feat+global_feat #B,N,D
        return feat




@DECODER_REGISTRY.register_module()
class pose_estimater(nn.Module):
    def __init__(self,input_dim=512,middle_dim=256,cat_num=6) -> None:
        super().__init__()
        self.mlp_r=MLP(input_dim,middle_dim,6)
        self.mlp_t=MLP(input_dim,middle_dim,3)
        self.mlp_s=MLP(input_dim,middle_dim,3)

    def forward(self,inst_feat,index):
        '''
        inst_faet:B,1,2*D
        ''' 
        inst_feat=inst_feat.squeeze(dim=1)
        r=self.mlp_r(inst_feat) #B,6
        t=self.mlp_t(inst_feat) #B,3
        s=self.mlp_s(inst_feat) #B,3

        r=self.Ortho6d2Mat(r[:,0:3],r[:,3:])

        return r,t,s
    def Ortho6d2Mat(self,x_raw, y_raw):
        y = self.normalize_vector(y_raw)
        z = self.cross_product(x_raw, y) #B,3
        z = self.normalize_vector(z)#B,3
        x = self.cross_product(y,z)#B,3

        x = x.unsqueeze(2)
        y = y.unsqueeze(2)
        z = z.unsqueeze(2)
        matrix = torch.cat((x,y,z),dim=2) #batch*3*3
        return matrix
    def normalize_vector(self, v, dim =1, return_mag =False):
        return torch.nn.functional.normalize(v,dim=dim)

    def cross_product(self,u, v):
        return torch.cross(u,v,dim=-1)
