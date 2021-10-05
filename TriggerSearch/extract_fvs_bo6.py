# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import os

import datasets
import numpy as np
import torch
import transformers
import json

import warnings
import utils_qa
warnings.filterwarnings("ignore")

import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import util.db as db
import time
import random

import math
import numpy as np


import round8_helper as helper

#Evaluate candidates using surrogate and pick candidates
def select_query(n,lmax,we,net=None,tokens=None,losses=None,pool=10000,l_=8):
    if net is None or tokens is None or losses is None or len(tokens)==0 or len(losses)==0:
        #return random triggers
        vocab_size=we.shape[0]-1;
        candidates=[];
        while len(candidates)<n:
            l=int(torch.LongTensor(1).random_(l_)+1);
            candidate=torch.LongTensor(l).random_(vocab_size);
            candidate=tuple(F.pad(candidate,(0,lmax-l),"constant",vocab_size).tolist());
            if not candidate in tokens and not candidate in candidates:
                candidates.append(candidate);
        
        return candidates
    else:
        #Generate a list of candidates (list of tuples)
        #tokens: list of tokens: niter x length
        #losses: observations ncontext x niter
        
        vocab_size=we.shape[0]-1;
        candidates=[];
        while len(candidates)<pool:
            l=int(torch.LongTensor(1).random_(l_)+1);
            candidate=torch.LongTensor(l).random_(vocab_size);
            candidate=tuple(F.pad(candidate,(0,lmax-l),"constant",vocab_size).tolist());
            if not candidate in tokens and not candidate in candidates:
                candidates.append(candidate);
        
        qxind=torch.LongTensor(candidates);
        
        #Generate vectors
        #Length goes into vector dimension
        losses=torch.stack(losses,dim=1)
        xind=torch.LongTensor(tokens);
        x=we[xind.view(-1),:].view(xind.shape[0],-1); 
        qx=we[qxind.view(-1),:].view(qxind.shape[0],-1); 
        ypreds=[];
        for i in range(losses.shape[0]):
            y=losses[i,:].view(-1,1)
            y=torch.log(y.clamp(min=1e-10,max=1e10));
            ypred=net(x,y,qx).data.view(-1);
            ypreds.append(ypred);
        
        ypreds=torch.stack(ypreds,dim=0);
        
        #Now compute objective
        s=ypreds.mean(dim=0);
        
        #Select lowest values for query
        _,ind=s.sort(dim=0);
        ind=ind.cpu().tolist();
        queries=[candidates[i] for i in ind[:n]];
        return queries;


#Evaluate candidates using surrogate and pick candidates
def select_query_min_gumbel(n,lmax,we,net=None,tokens=None,losses=None,pool=10000):
    
    vocab_size=we.shape[0]-1;
    candidates=[];
    xind=torch.LongTensor(tokens);
    nqueries=xind.shape[0];
    x=we[xind.view(-1),:].view(xind.shape[0],-1); 
    losses=torch.stack(losses,dim=1)
    
    ncontext=losses.shape[0];
    xs=x.view(1,nqueries,-1).repeat(ncontext,1,1);
    ys=losses.view(ncontext,nqueries,1);
    candidates=net.find_min_gumbel(xs,ys,we,l=lmax);
    
    candidates=[candidates[i].clone() for i in range(candidates.shape[0])];
    
    return candidates;


#Evaluate candidates using surrogate and pick candidates
def select_query_min(n,lmax,we,net=None,tokens=None,losses=None,pool=10000,l_=8):
    
    vocab_size=we.shape[0]-1;
    candidates=[];
    xind=torch.LongTensor(tokens);
    x=we[xind.view(-1),:].view(xind.shape[0],-1); 
    losses=torch.stack(losses,dim=0)
    losses=torch.log(losses.clamp(min=1e-20,max=1e20)).mean(dim=1,keepdim=True);#Integrate losses into 1
    best_scores=[];
    
    
    y=losses.view(-1,1).cuda();
    #y=torch.log(y.clamp(min=1e-10,max=1e10));
    candidate,best_scores_i=net.find_min_v2(x,y,we,l=l_,dupes=xind);
    candidate=[j.data.view(-1) for j in candidate]
    #candidate=[(6882,)];
    candidates=candidates+candidate;
    best_scores=best_scores+best_scores_i
    
    print(sum(best_scores)/len(best_scores))
    return candidates;




def extract_bo_features(model,tokenizer,dataset):
    batch_size=12
    
    #Surrogate
    arch='arch.maml_surrogate_linear_gp2_multi'
    nh=512
    nlayers=4
    checkpoint='linear_512_900_v2.pt'
    
    #Optimization
    maxl=8
    actual_l=6;
    num_queries_per_iter=actual_l
    niter=210;
    pool=3000;
    
    
    we=helper.get_we(model).clone();
    ids=helper.text2tokens(tokenizer,'? . ,');
    #Create an "empty token" index
    we=F.pad(we,(0,1,0,1));
    we[-1,-1]=1;
    
    
    #Load data
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size);
    losses=[];
    for data in dataloader:
        break;
    
    #Load surrogate
    import importlib
    neural_process=importlib.import_module(arch)
    surrogate=neural_process.new(nh=nh,nlayers=nlayers,length=maxl).cuda();
    try:
        checkpoint=torch.load(checkpoint);
    except:
        checkpoint=torch.load(os.path.join('/',checkpoint));
    
    surrogate.load_state_dict(checkpoint);
    surrogate.float();
    surrogate.register_we(we);
    #surrogate=None;
    
    #Run optimization
    tokens=[];
    losses=[];
    fvs=[];
    t0=time.time();
    with torch.no_grad():
        for iter in range(niter):
            if iter<20:
                queries=select_query(num_queries_per_iter,maxl,we,tokens=tokens,l_=actual_l);
            else:
                queries=select_query_min(num_queries_per_iter,maxl,we,surrogate,tokens,losses,pool=pool,l_=actual_l);
            
            print('time %.2f'%(time.time()-t0));
            #Evaluate queries over contexts
            for q in queries:
                ind=torch.LongTensor([i for i in q if i<we.shape[0]-1]);
                trigger=we[ind,:-1].view(1,len(ind),-1);
                losses_q=[];
                for start_idx in [5,25,45]:
                    loss_trigger_i,_,_=helper.forward_embed(model,data,trigger=trigger,start_idx=start_idx);
                    loss_trigger_i=-torch.log(-torch.expm1(-loss_trigger_i.data));
                    losses_q.append(loss_trigger_i.view(-1));
                
                losses_q=torch.cat(losses_q,dim=0);
                
                tokens.append(q);
                losses.append(losses_q);
            
            min_loss=torch.log(torch.stack(losses,dim=0).clamp(min=1e-20,max=1e20)).mean(dim=1).min();
            print('iter %d, loss %.2f, size %d (%d), time %.2f'%(iter,min_loss,len(losses),len(set(tokens)),time.time()-t0));
            sys.stdout.flush();
    
    tokens=torch.LongTensor(tokens);
    fvs=torch.stack(losses,dim=0).data.cpu(); # niter , ncontext (start_idx x batch_size)
    print(fvs.shape);
    return fvs,tokens;



def extract_fvs_(id=0,root='data/round8-train-dataset/models'):
    id='id-%08d'%id;
    f=open(os.path.join(root,id,'config.json'),'r');
    config=json.load(f);
    f.close();
    
    model_filepath=os.path.join(root,id,'model.pt');
    #examples_dirpath=os.path.join(root,id,'example_data');
    examples_dirpath='selected_data.json';
    scratch_dirpath='./scratch'
    if 'electra' in config['model_architecture']:
        tokenizer_filepath='data/round8-train-dataset/tokenizers/tokenizer-google-electra-small-discriminator.pt';
    elif 'squad2' in config['model_architecture']:
        tokenizer_filepath='data/round8-train-dataset/tokenizers/tokenizer-deepset-roberta-base-squad2.pt';
    elif 'roberta' in config['model_architecture']:
        tokenizer_filepath='data/round8-train-dataset/tokenizers/tokenizer-roberta-base.pt';
    else:
        a=0/0;
    
    return extract_fvs(model_filepath, tokenizer_filepath, scratch_dirpath, examples_dirpath);

def extract_fvs(model_filepath, tokenizer_filepath, scratch_dirpath, examples_dirpath):
    model = torch.load(model_filepath).cuda()
    model.eval()
    if not examples_dirpath.endswith('json'):
        fns = [fn for fn in os.listdir(examples_dirpath) if fn.endswith('.json')]
        fns.sort()
        assert fns[0].find('poison')<0
        fns=[os.path.join(examples_dirpath, fn) for fn in fns]
        examples_filepath = fns[0]
    else:
        examples_filepath=examples_dirpath
    
    dataset = datasets.load_dataset('json', data_files=[examples_filepath], field='data', keep_in_memory=True, split='train', cache_dir=os.path.join(scratch_dirpath, '.cache'))
    
    tokenizer = torch.load(tokenizer_filepath)
    tokenized_dataset = helper.tokenize_for_qa(tokenizer, dataset)
    tokenized_dataset.set_format('pt', columns=['input_ids', 'attention_mask', 'token_type_ids', 'start_positions', 'end_positions'])
    
    fvs,tokens=extract_bo_features(model,tokenizer,tokenized_dataset);
    
    print(fvs.shape,tokens.shape)
    return fvs,tokens;
    
if __name__ == "__main__":
    data=db.Table({'model_id':[],'label':[],'model_name':[],'fvs_surrogate':[],'tokens_surrogate':[]});
    data=db.DB({'table_ann':data});
    root='data/round8-train-dataset/models'
    t0=time.time()
    
    model_ids=list(range(0,120)) #96
    
    for i,id in enumerate(model_ids):
        print(i,id)
        fvs,tokens=extract_fvs_(id,root=root);
        
        f=open(os.path.join(root,'id-%08d'%id,'config.json'),'r');
        config=json.load(f);
        f.close();
        
        fname=os.path.join(root,'id-%08d'%id,'ground_truth.csv');
        f=open(fname,'r');
        for line in f:
            line.rstrip('\n').rstrip('\r')
            label=int(line);
            break;
        
        f.close();
        
        data['table_ann']['model_name'].append('id-%08d'%id);
        data['table_ann']['model_id'].append(id);
        data['table_ann']['label'].append(label);
        data['table_ann']['fvs_surrogate'].append(fvs);
        data['table_ann']['tokens_surrogate'].append(tokens);
        print('Model %d(%d), time %f'%(i,id,time.time()-t0));
        
        data.save('data_r8_surrogate6_linear_512_900_v2.pt');
        #data.save('tmp.pt');
        
