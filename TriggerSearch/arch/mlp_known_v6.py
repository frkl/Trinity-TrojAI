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



class new(nn.Module):
    def __init__(self,params):
        super(new,self).__init__()
        nh=params.nh;
        nh2=params.nh2
        
        self.interval=nh
        n=int((540-1)/nh)+1
        
        self.encoder2=MLP(3*12*3,nh2,2,params.nlayers2);
        self.w=nn.Parameter(torch.Tensor(1).fill_(1));
        self.b=nn.Parameter(torch.Tensor(1).fill_(0));
        self.margin=params.margin;
        return;
    
    def forward(self,data_batch):
        grads=data_batch['fvs_surrogate'];
        b=len(grads);
        
        hs=[];
        #Have to process one by one due to variable nim & nclasses
        for i in range(b):
            x=data_batch['fvs_surrogate'][i].cuda();
            #print(x.shape)
            x=torch.log(x.clamp(min=1e-20,max=1e20));
            x=x.view(-1,3*12);
            x,_=torch.cummin(x,dim=0);
            #x=x.mean(dim=-1);
            #x=x.view(-1);
            #x,_=x.sort(dim=0)
            #ind=[0,59,251];
            h=x[[0,119,1249],:].clone() #50
            h=h.view(-1);
            h=h.unsqueeze(0);
            #print(h.shape)
            hs.append(h);
        
        h=torch.cat(hs,dim=0);
        h=self.encoder2(h);
        h=torch.tanh(h)*self.margin;
        return h
    
    def logp(self,data_batch):
        h=self.forward(data_batch);
        return h[:,1]-h[:,0];