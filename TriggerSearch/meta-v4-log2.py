
#import torch
#from performer_pytorch import Performer

#model=Performer(dim=256,dim_head=256,depth=12,heads=8).cuda();

#x=torch.Tensor(1,2048,256).requires_grad_().cuda(); #
#y=model(x);
#
#y.mean().backward()

#print(y.shape)


#Learn a token_gen using meta-learning
#Such that obj decrease is accelerated on given data

#Python2,3 compatible headers
from __future__ import unicode_literals,division
from builtins import int
from builtins import range

#System packages
import torch
from torch.autograd import Variable,grad
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy
import scipy
import scipy.misc
import math
import time
import random
import argparse
import sys
import os
import re
import copy
import importlib
import json
from collections import namedtuple
from collections import OrderedDict
from itertools import chain
import util.db as db

import os
import numpy as np
import copy
import torch
import transformers

import warnings
warnings.filterwarnings("ignore")
import util.db as db

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import random
from functools import partial

import util.session_manager as session_manager
import util.smartparse as smartparse
import util.file
import pandas

import round8_helper as helper

#torch.set_default_dtype(torch.float64)

# Training settings
def default_params():
    params=smartparse.obj();
    #Data
    params.ninput=129*8;
    params.nh=512;
    params.nlayers=4;
    params.lr=1e-3;
    params.load='';
    params.arch='arch.maml_surrogate_linear_gp_multi';
    params.session_dir=None;
    
    params.maxl=8;
    return params


def create_session(params):
    session=session_manager.Session(session_dir=params.session_dir); #Create session
    torch.save({'params':params},session.file('params.pt'));
    pmvs=vars(params);
    pmvs=dict([(k,pmvs[k]) for k in pmvs if not(k=='stuff')]);
    print(pmvs);
    util.file.write_json(session.file('params.json'),pmvs); #Write a human-readable parameter json
    session.file('model','dummy');
    return session;

params = smartparse.parse()
params = smartparse.merge(params, default_params())
params.argv=sys.argv;
session=create_session(params);
params.session=session;





train_params=torch.Tensor(1000,3).uniform_(-3.14,3.14);
test_params=torch.Tensor(300,3).uniform_(-3.14,3.14);

class trajectories:
    def __init__(self,inds):
        data=[];
        we=[];
        meta=pandas.read_csv('data/round8-train-dataset/METADATA.csv');
        
        for i in inds:
            #Remove mobilebert at the moment
            id=list(meta['model_name']).index('id-%08d'%i);
            #if not(meta['model_architecture'][id]=='google/electra-small-discriminator'):
            #    continue;
            try:
                data_i=torch.load('meta-r8-sel2/%d.pt'%i);
            except:
                continue;
            
            #Get input embedding matrix
            paths=helper.get_paths(i);
            model,tokenizer,dataset=helper.load_stuff(*paths)
            we_i=helper.get_we(model);
            we_i=we_i.data.cpu()#.double();
            #Create an "empty token" index
            we_i=F.pad(we_i,(0,1,0,1));
            we_i[-1,-1]=1;
            
            #Compute actual input sequences
            scores_i=torch.log(data_i['loss'].clamp(min=1e-20,max=20))#.double()
            
            ind=[];
            xind=[];
            L=params.maxl;
            for j,trigger in enumerate(data_i['triggers']):
                if len(trigger)<=L:
                    x=F.pad(torch.LongTensor(trigger),(0,L-len(trigger)),"constant",we_i.shape[0]-1);
                    xind.append(x);
                    ind.append(j)
            
            xind=torch.stack(xind,dim=0);
            #we_i=we_i[xind.view(-1),:].view(xind.shape[0],xind.shape[1]*we_i.shape[1])
            we.append(we_i)
            print(we_i.shape)
            scores_i=scores_i[:,:,torch.LongTensor(ind)].clone();
            
            for j in range(scores_i.shape[0]):
                for k in range(scores_i.shape[1]):
                    x=xind.clone().cpu();
                    y=scores_i.data[j,k,:].clone().cpu()
                    data.append({'x':x,'y':y,'we_ind':len(we)-1});
        
        self.data=data;
        self.we=we;
    
    def __len__(self):
        return len(self.data);
    
    def __getitem__(self,i):
        data=self.data[i];
        xind=data['x'];
        y=data['y'];
        we_ind=data['we_ind'];
        we=self.we[we_ind];
        
        Ntrain=int(torch.LongTensor(1).random_(3000)+300);
        Ntest=1000;
        Nx=30;
        assert len(y)>=Ntrain+Ntest;
        ind=torch.randperm(len(y))[:Ntrain+Ntest];
        
        xtrain=xind[ind[:Ntrain],:];
        xtrain=we[xtrain].view(xtrain.shape[0],-1).clone().cuda();
        ytrain=y[ind[:Ntrain]].clone().cuda().view(-1,1);
        
        xtest=xind[ind[Ntrain:],:];
        xtest=we[xtest].view(xtest.shape[0],-1).clone().cuda();
        ytest=y[ind[Ntrain:]].clone().cuda().view(-1,1);
        
        
        #Top/bottom 100
        _,ind=y.sort(dim=0);
        ind=ind[:Nx].tolist()+ind[-Nx:].tolist();
        ind=torch.LongTensor(ind);
        
        xex=xind[ind,:];
        xex=we[xex].view(xex.shape[0],-1).clone().cuda();
        yex=y[ind].clone().cuda().view(-1,1);
        
        return xtrain,ytrain,xtest,ytest,xex,yex;
    
    def all(self,i):
        scores=self.data[i];
        we=self.we[self.we_ind[i]];
        x=we.clone().cuda();
        y=scores.view(-1,1).clone().cuda();
        return x,y


ready_models=range(0,120);
n=len(ready_models)//2;
train_dset=trajectories(ready_models[:n]);
test_dset=trajectories(ready_models[n:]);
print('Loaded %d train %d test'%(len(train_dset),len(test_dset)))

#import arch.neural_process_surrogate as neural_process
#import arch.neural_process_surrogate_bert as neural_process
#import arch.neural_process_surrogate_maml_rr_batch as neural_process
import importlib
neural_process=importlib.import_module(params.arch)
#import arch.neural_process_surrogate_z as neural_process
surrogate=neural_process.new(params.ninput,nh=params.nh,nlayers=params.nlayers,length=params.maxl).cuda();
if not params.load=='':
    checkpoint=torch.load(params.load);
    surrogate.load_state_dict(checkpoint);

opt=optim.Adamax(surrogate.parameters(),lr=params.lr);

colors=['r','g','b','c','m','y','k','darkorange','greenyellow','gray','pink','darkgoldenrod']
def plot_sequences(X,fname):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    #Create scatter plots by data type
    for i in range(len(X)):
        x=X[i][0].cpu().numpy()
        y=X[i][1].cpu().numpy()
        ax.plot(x,y,'-o',label='%d'%i,c=colors[i]);
        #for j in range(len(x)):
        #    ax.annotate('%d'%j,(x[j], y[j]))
    
    lgd=plt.legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=1)
    #ax.legend();
    plt.savefig(fname, bbox_extra_artists=[lgd], bbox_inches='tight');
    plt.close(fig)



batch=16;
t0=time.time();
for epoch in range(1000000):
    if epoch%10==99:
        surrogate.eval()
        acc=[];
        acc_random=[];
        for i in range(len(test_dset)):
            x,y=test_dset.all(i);
            min_global,min_found,min_random=find_min(surrogate,x,y);
            if min_global<=-2.3:
                acc.append(float(min_found<-2.3));
                acc_random.append(float(min_random<-2.3));
        
        session.log('min %f, rng %f'%(sum(acc)/len(acc),sum(acc_random)/len(acc_random)));
    
    trainloss=[];
    trainloss_ex=[];
    opt.zero_grad();
    ind=torch.randperm(len(train_dset));
    surrogate.train()
    for i in range(len(train_dset)):
        print('%d/%d      '%(i,len(train_dset)),end='\r');
        id=ind[i];
        x0,y0,x1,y1,xex,yex=train_dset[id];
        
        pred_y1=surrogate(x0,y0,x1);
        diff=y1-pred_y1;
        loss_i=(diff**2).mean();
        
        pred_yex=surrogate(x0,y0,xex);
        diff_ex=yex-pred_yex;
        loss_ex_i=(diff_ex**2).mean();
        
        loss=loss_i+loss_ex_i;
        loss.backward();
        
        trainloss.append(float(loss_i));
        trainloss_ex.append(float(loss_ex_i));
        
        if (i+1)%batch==0:
            opt.step();
            opt.zero_grad();
    
    trainloss=sum(trainloss)/len(trainloss);
    trainloss_ex=sum(trainloss_ex)/len(trainloss_ex);
    
    if epoch%10==0:
        testloss=[];
        testloss_ex=[];
        acc=[];
        acc_random=[];
        surrogate.eval()
        for i in range(len(test_dset)):
            x0,y0,x1,y1,xex,yex=test_dset[i];
            
            pred_y1=surrogate(x0,y0,x1);
            diff=y1-pred_y1;
            loss_i=(diff**2).mean();
            
            pred_yex=surrogate(x0,y0,xex);
            diff_ex=yex-pred_yex;
            loss_ex_i=(diff_ex**2).mean();
            
            loss=loss_i+loss_ex_i;
            loss_i.backward();
            
            testloss.append(float(loss_i));
            testloss_ex.append(float(loss_ex_i));
        
        
        
        testloss=sum(testloss)/len(testloss);
        testloss_ex=sum(testloss_ex)/len(testloss_ex);
    
    session.log('Epoch %d, train %f - %f, test %f - %f, time %.2f'%(epoch,trainloss,trainloss_ex,testloss,testloss_ex,time.time()-t0));
    
    
    
    
    if epoch%100==0:
        torch.save(surrogate.state_dict(),session.file('model','%d.pt'%epoch));
    
    

    
    
    

