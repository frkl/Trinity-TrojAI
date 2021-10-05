import torch
import torchvision.models
from torch.autograd import grad
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import WeightNorm
import math
from collections import OrderedDict as OrderedDict
import copy


class MLP(nn.Module):
    def __init__(self,ninput,nh,noutput,nlayers):
        super().__init__()
        
        self.layers=nn.ModuleList();
        self.bn=nn.LayerNorm(ninput);
        if nlayers==1:
            self.layers.append(nn.Linear(ninput,noutput));
        else:
            self.layers.append(nn.Linear(ninput,nh));
            for i in range(nlayers-2):
                self.layers.append(nn.Linear(nh,nh));
            
            self.layers.append(nn.Linear(nh,noutput));
        
        self.ninput=ninput;
        self.noutput=noutput;
        return;
    
    def forward(self,x):
        h=x.view(-1,self.ninput);
        #h=self.bn(h);
        for i in range(len(self.layers)-1):
            h=self.layers[i](h);
            h=F.relu(h);
            #h=F.dropout(h,training=self.training);
        
        h=self.layers[-1](h);
        h=h.view(*(list(x.shape[:-1])+[-1]));
        return h


from performer_pytorch import Performer

class MLP_lowmem(nn.Module):
    def __init__(self,ninput,nh,noutput,nlayers):
        super(MLP_lowmem,self).__init__()
        self.net=MLP(ninput,nh,noutput,nlayers);
    
    def forward(self,x,batch=8):
        h=torch.utils.checkpoint.checkpoint(self.net,x.requires_grad_());
        return h;


class Performer_lowmem(nn.Module):
    def __init__(self,dim,dim_head,depth,heads):
        super(Performer_lowmem,self).__init__()
        self.pos=nn.Parameter(torch.Tensor(2048,dim).normal_()*0.1);
        self.net=Performer(dim=dim,dim_head=dim_head,depth=depth,heads=heads);
    
    def forward(self,x,batch=8):
        N=x.shape[0];
        x=x+self.pos.unsqueeze(0)[:,:x.shape[1],:];
        hs=[];
        for i in range(0,N,batch):
            r=min(i+batch,N);
            h=torch.utils.checkpoint.checkpoint(self.net,x[i:r]);
            hs.append(h);
        
        hs=torch.cat(hs,dim=0);
        return hs;



class new(nn.Module):
    def __init__(self,params):
        super(new,self).__init__()
        nh=params.nh;
        nh2=params.nh2
        
        self.encoder1=MLP(512,nh,nh,params.nlayers);
        self.encoder2=MLP(nh,nh2,2,params.nlayers2);
        self.w=nn.Parameter(torch.Tensor(1).fill_(1));
        self.b=nn.Parameter(torch.Tensor(1).fill_(0));
        self.margin=params.margin;
        return;
    
    def forward(self,data_batch):
        grads=data_batch['ws_surrogate'];
        b=len(grads);
        
        hs=[];
        #Have to process one by one due to variable nim & nclasses
        for i in range(b):
            x=data_batch['ws_surrogate'][i].cuda();
            h=self.encoder1(x).mean(0,keepdim=True);
            hs.append(h);
        
        h=torch.cat(hs,dim=0);
        h=self.encoder2(h);
        h=torch.tanh(h)*self.margin;
        return h
    
    def logp(self,data_batch):
        h=self.forward(data_batch);
        return h[:,1]-h[:,0];