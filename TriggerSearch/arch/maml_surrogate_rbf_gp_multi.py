import torch
import torch.linalg
import torchvision.models
from torch.autograd import grad
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import WeightNorm
import math
from collections import OrderedDict as OrderedDict
import copy
import time


import torch.optim as optim

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
        
        self.pre=False;
        self.embeddings=[];
        
        return;
    
    def forward(self,x):
        if self.pre:
            assert len(self.layers)>=2
            #x would be length L list of integers,
            e=[];
            for i in range(len(x)):
                if isinstance(x[i],int):
                    e_i=self.embedding[i][x[i]:x[i]+1,:];
                else:
                    e_i=self.embedding[i][x[i].view(-1),:];
                e.append(e_i);
            
            e_=e[0];
            for i in range(1,len(x)):
                e_=e_+e[i];
            
            e_=e_+self.layers[0].bias.data.view(1,-1);
            h=e_;
            h=F.relu(h);
            for i in range(1,len(self.layers)-1):
                h=self.layers[i](h);
                h=F.relu(h);
                #h=F.dropout(h,training=self.training);
            
            h=self.layers[-1](h);
            return h
        
        
        
        h=x.view(-1,self.ninput);
        #h=self.bn(h);
        for i in range(len(self.layers)-1):
            h=self.layers[i](h);
            h=F.relu(h);
            #h=F.dropout(h,training=self.training);
        
        h=self.layers[-1](h);
        h=h.view(*(list(x.shape[:-1])+[-1]));
        return h
    
    def pre_multiply(self,we):
        nh=we.shape[1];
        n=self.layers[0].weight.shape[1]//nh;
        #Convert layer 0 into embeddings
        self.pre=True;
        self.embedding=[];
        for i in range(n):
            e=torch.mm(we,self.layers[0].weight.data[:,i*nh:(i+1)*nh].t());
            self.embedding.append(e.data);
        
        return;
        
    def reset(self):
        self.pre=False;


#def kernel(x,x2):
#    if len(x.shape)==3:
#        d=torch.bmm(x,x2.permute(0,2,1));
#    else:
#        d=torch.mm(x,x2.permute(1,0));
#    return d


class new(nn.Module):
    def __init__(self,ninput=768,nh=512,nlayers=3,length=8):
        super(new,self).__init__()
        self.params=nn.ParameterList();
        self.encoder1=MLP(129*length,nh,nh,nlayers).double();
        self.encoder2=MLP(769*length,nh,nh,nlayers).double();
        self.reg=nn.Parameter(torch.Tensor(1).fill_(0).double());
        self.s=nn.Parameter(torch.Tensor(1).fill_(0).double());
        self.length=length;
        return;
    
    def kernel(self,x,x2,s=1.0):
        d=torch.cdist(x,x2,p=2.0)**2;
        s=0.5*torch.exp(-2*self.s);
        K=torch.exp(-s*d)
        return K
    
    def kernel_z(self,x):
        return 1;
    
    def embed(self,x):
        if x.shape[-1]==129*self.length:
            e=self.encoder1(x);
        elif x.shape[-1]==769*self.length:
            e=self.encoder2(x);
        else:
            print(x.shape)
            a=0/0
        
        return e;
    
    def regress(self,xs,ys):
        dtype=copy.deepcopy(xs.dtype)
        xs=xs.type(self.reg.data.dtype)
        ys=ys.type(self.reg.data.dtype)
        d=3;
        if len(xs.shape)==2:
            d=2;
            xs=xs.unsqueeze(0)
            ys=ys.unsqueeze(0)
        
        reg=torch.exp(self.reg);
        e=self.embed(xs);
        N=xs.shape[1];
        nh=e.shape[-1];
        
        b=ys.mean(1,keepdim=True); #1x1
        K=self.kernel(e,e);
        A=K+reg*torch.eye(N).to(e.device).unsqueeze(0); #NxN
        A=torch.inverse(A);
        a=torch.bmm(A,ys-b); #Nx1
        S=torch.eye(N).to(A.device).unsqueeze(0)-A; #NxN
        
        if d==2:
            e=e.squeeze(0);
            a=a.squeeze(0);
            b=b.squeeze(0);
            A=A.squeeze(0);
            S=S.squeeze(0);
        
        return e,a,b,A
    
    def score(self,qxs,e,a,b,S):
        dtype=copy.deepcopy(qxs.dtype)
        qxs=qxs.type(self.reg.data.dtype)
        
        d=3;
        if len(qxs.shape)==2:
            d=2;
            e=e.unsqueeze(0);
            a=a.unsqueeze(0);
            b=b.unsqueeze(0);
            S=S.unsqueeze(0);
            qxs=qxs.unsqueeze(0);
        
        e2=self.embed(qxs);
        K2=self.kernel(e2,e);
        
        qys=torch.bmm(K2,a)+b; #Nx1
        qys_var=(torch.bmm(K2,S)*K2).sum(2,keepdim=True);
        qys_std=qys_var.clamp(min=1e-20)**0.5;
        
        if d==2:
            qys=qys.squeeze(0);
            qys_std=qys_std.squeeze(0);
        
        qys=qys.type(dtype);
        qys_std=qys_std.type(dtype);
        return qys,qys_std;
    
    
    def forward(self,xs,ys,qxs,niter=5,uncertainty=False):
        e,a,b,S=self.regress(xs,ys);
        qys,qys_std=self.score(qxs,e,a,b,S);
        
        if uncertainty:
            return qys,qxs;
        else:
            return qys;
    
    def find_min_v2(self,xs,ys,we,l=8,dupes=[]):
        if len(dupes)>0:
            dupes=torch.LongTensor(dupes).cuda();
        
        e,a,b,S=self.regress(xs,ys);
        best_scores=[];
        if xs.shape[1]==129*self.length:
            self.encoder1.pre_multiply(we);
        elif xs.shape[1]==769*self.length:
            self.encoder2.pre_multiply(we);
        
        current_best=ys.min();
        
        def check_dupe(tokens,ind,dupes,vocab_size):
            l=len(tokens);
            q=tokens.clone().to(dupes.device);
            q[ind]=-1;
            #Identify available indicies
            avail=list(range(vocab_size-1));
            if len(dupes)>0:
                match=(dupes-q.view(1,-1)).eq(0).long().sum(dim=1).eq(l-1).nonzero();
                match=match.view(-1).tolist()
                if len(match)>0:
                    dupe_inds=[int(dupes[j][ind]) for j in match];
                    avail=list(set(avail).difference(set(dupe_inds)));
            
            return avail;
            
        
        with torch.no_grad():
            nhword=we.shape[1];
            vocab_size=we.shape[0];
            tokens=torch.LongTensor(self.length).fill_(vocab_size-1);
            candidates=[];
            for i in range(l):
                best_score=1e10;
                tokens[i]=int(torch.LongTensor(1).random_(vocab_size-1));
                while True:
                    improved=False
                    for ind in range(i,-1,-1):
                        avail=check_dupe(tokens,ind,dupes,vocab_size);
                        
                        tokens_=tokens.tolist();
                        tokens_[ind]=torch.LongTensor(avail).cuda();
                        
                        #Could not use modular embed due to special format
                        if xs.shape[1]==129*self.length:
                            e2=self.encoder1(tokens_);
                        elif xs.shape[1]==769*self.length:
                            e2=self.encoder2(tokens_);
                        
                        #Compute prediction and std
                        K2=self.kernel(e2,e);
                        Kz=self.kernel_z(e2);
                        
                        qy=torch.mm(K2,a)+b; #Nx1
                        qy_var=Kz-(torch.mm(K2,S)*K2).sum(dim=-1,keepdim=True);
                        qy_std=qy_var.clamp(min=1e-20)**0.5;
                        
                        #Compute expected improvement
                        s=(qy-current_best)/qy_std;
                        #print(qy.min(),current_best,qy_std.mean())
                        
                        score,j=s.min(dim=0);
                        score=float(score);
                        j=int(tokens_[ind][j]);
                        if score<best_score:
                            best_score=score
                            tokens[ind]=j;
                            improved=True;
                    
                    #print('length %d, score %.4f'%(i,best_score));
                    if not improved:
                        break;
                
                candidates.append(tokens.clone());
                best_scores.append(float(score));
        
        self.encoder1.reset();
        self.encoder2.reset();
        return candidates,best_scores
    
    def find_min_v3(self,xs,ys,we,l=8,dupes=[],l_=1):
        if len(dupes)>0:
            dupes=torch.LongTensor(dupes).cuda();
        
        w=self.embedw(xs,ys).view(-1,1);
        best_scores=[];
        
        if xs.shape[1]==129*8:
            self.encoder1.pre_multiply(we);
        elif xs.shape[1]==769*8:
            self.encoder2.pre_multiply(we);
        
        with torch.no_grad():
            nh=we.shape[1];
            vocab_size=we.shape[0];
            tokens=torch.LongTensor(l).fill_(vocab_size-1);
            tokens_l=[];
            for i in range(l_):
                best_score=1e10;
                tokens[i]=int(torch.LongTensor(1).random_(vocab_size-1));
                while True:
                    improved=False
                    for ind in range(i,-1,-1):
                        avail=list(range(vocab_size-1));
                        if len(dupes)>0:
                            q=tokens.clone().cuda();
                            q[ind]=-1;
                            match=(dupes-q.view(1,-1)).eq(0).sum(dim=1).eq(l-1).nonzero();
                            match=match.view(-1).tolist()
                            if len(match)>0:
                                #print('dup %d'%len(match))
                                dupe_inds=[int(dupes[j][ind]) for j in match];
                                avail=list(set(avail).difference(set(dupe_inds)));
                        
                            
                        tokens_=tokens.tolist();
                        tokens_[ind]=torch.LongTensor(avail).cuda();
                        if xs.shape[1]==129*self.length:
                            e=self.encoder1(tokens_);
                        elif xs.shape[1]==769*self.length:
                            e=self.encoder2(tokens_);
                        s=torch.mm(e,w).view(-1);
                        score,j=s.min(dim=0);
                        score=float(score);
                        j=int(tokens_[ind][j]);
                        if score<best_score:
                            best_score=score
                            tokens[ind]=j;
                            improved=True;
                    
                    #print('length %d, score %.4f'%(i,best_score));
                    if not improved:
                        break;
                
                tokens_l.append(tokens.clone());
                best_scores.append(float(score));
        
        self.encoder1.reset();
        self.encoder2.reset();
        return tokens_l,best_scores
    