
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
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import util.db as db
import time
import random

import math
import numpy as np


import round8_helper as helper


t0=time.time();
def extract_responses(model,tokenizer,dataset,trigger_gt):
    we=helper.get_we(model).data.clone();
    #print('Nex %d'%len(dataset),we.shape);
    #return None;
    
    #Enumerate length 1
    #triggers=[];
    #for i in range(1,we.shape[0]):
    #    trigger=tuple([i])
    #    triggers.append(trigger);
    
    
    #Produce 30k triggers
    triggers=[];
    ntrig=50000
    maxl=8;
    if not trigger_gt is None:
        trigger_gt=helper.text2tokens(tokenizer,trigger_gt);
        #Insert GT triggers
        for s in range(0,len(trigger_gt)):
            for t in range(s+1,min(s+maxl,len(trigger_gt))):
                triggers.append(tuple(trigger_gt[s:t]));
    
    while len(triggers)<ntrig:
        l=int(torch.LongTensor(1).random_(maxl)+1);
        print('%d   '%len(triggers),end='\r')
        while True:
            trigger=torch.LongTensor(l).random_(we.shape[0]).tolist();
            trigger=tuple(trigger);
            if not trigger in triggers:
                triggers.append(trigger);
                break;
    
    triggers.reverse();
    
    batch_size=20;
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size);
    batches=[];
    for data in dataloader:
        batches.append(data);
    
    L=batches[0]['input_ids'].shape[1];
    assert L>45
    
    losses=[];
    for start_idx in [5,25,45]:
        losses_d=[];
        for i,wid in enumerate(triggers):
            ind=torch.LongTensor(wid).view(-1);
            trigger=we[ind,:].view(1,len(wid),-1);
            with torch.no_grad():
                loss_trigger_i=[];
                for data in batches:
                    l,_,_=helper.forward_embed(model,data,trigger=trigger,start_idx=start_idx);
                    loss_trigger_i.append(l);
                loss_trigger_i=torch.cat(loss_trigger_i,dim=0);
                loss_trigger_i=-torch.log(-torch.expm1(-loss_trigger_i));
            
            losses_d.append(loss_trigger_i.data.cpu());
            print('%d %d/%d, time %.2f      '%(start_idx,i,len(triggers),time.time()-t0),end='\r')
        
        losses_d=torch.stack(losses_d,dim=1);
        losses.append(losses_d)
    
    losses=torch.stack(losses,dim=1); # batch x idx x triggers
    
    return we,triggers,losses;


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
    
    #if ('electra' in config['model_architecture']):
    #    return None,None,None;
    
    try:
        trigger=config['trigger']['trigger_executor']['trigger_text'];
    except:
        trigger=None;
    
    return extract_fvs(model_filepath, tokenizer_filepath, scratch_dirpath, examples_dirpath,trigger);

def extract_fvs(model_filepath, tokenizer_filepath, scratch_dirpath, examples_dirpath,trigger):
    model = torch.load(model_filepath).cuda()
    model.eval()
    
    if examples_dirpath.endswith('.json'):
        examples_filepath=examples_dirpath
    else:
        fns = [fn for fn in os.listdir(examples_dirpath) if fn.endswith('.json')]
        fns.sort()
        assert fns[0].find('poison')<0
        fns=[os.path.join(examples_dirpath, fn) for fn in fns]
        examples_filepath = fns[0]
    
    dataset = datasets.load_dataset('json', data_files=[examples_filepath], field='data', keep_in_memory=True, split='train', cache_dir=os.path.join(scratch_dirpath, '.cache'))
    
    tokenizer = torch.load(tokenizer_filepath)
    tokenized_dataset = helper.tokenize_for_qa(tokenizer, dataset)
    tokenized_dataset.set_format('pt', columns=['input_ids', 'attention_mask', 'token_type_ids', 'start_positions', 'end_positions'])
    
    return extract_responses(model,tokenizer,tokenized_dataset,trigger);
    

data=db.Table({'model_id':[],'label':[],'model_name':[],'fvs_known':[]})
data=db.DB({'table_ann':data});
root='data/round8-train-dataset/models'
t0=time.time()

model_ids=list(range(0,120))
#random.shuffle(model_ids)

for i,id in enumerate(model_ids):
    we,triggers,losses=extract_fvs_(id,root=root);
    if not we is None:
        torch.save({'triggers':triggers,'loss':losses},'meta-r8-sel2/%d.pt'%id);
    print('Model %d(%d), time %f'%(i,id,time.time()-t0));


