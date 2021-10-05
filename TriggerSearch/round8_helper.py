


import os

import datasets
import numpy as np
import torch
import transformers
import json

import warnings

import utils_qa

import torch.nn as nn
import torch.nn.functional as F

def get_paths(id,root='data/round8-train-dataset/models'):
    id='id-%08d'%id;
    f=open(os.path.join(root,id,'config.json'),'r');
    config=json.load(f);
    f.close();
    
    model_filepath=os.path.join(root,id,'model.pt');
    examples_dirpath=os.path.join(root,id,'example_data');
    scratch_dirpath='./scratch'
    if 'electra' in config['model_architecture']:
        tokenizer_filepath='data/round8-train-dataset/tokenizers/tokenizer-google-electra-small-discriminator.pt';
    elif 'squad2' in config['model_architecture']:
        tokenizer_filepath='data/round8-train-dataset/tokenizers/tokenizer-deepset-roberta-base-squad2.pt';
    elif 'roberta' in config['model_architecture']:
        tokenizer_filepath='data/round8-train-dataset/tokenizers/tokenizer-roberta-base.pt';
    else:
        a=0/0;
    
    return model_filepath, tokenizer_filepath, scratch_dirpath, examples_dirpath;

def load_stuff(model_filepath, tokenizer_filepath, scratch_dirpath, examples_dirpath):
    model = torch.load(model_filepath).cuda()
    model.eval()
    
    if examples_dirpath.endswith('.json'):
        examples_filepath=examples_dirpath;
    else:
        fns = [fn for fn in os.listdir(examples_dirpath) if fn.endswith('.json')]
        fns.sort()
        assert fns[0].find('poison')<0
        fns=[os.path.join(examples_dirpath, fn) for fn in fns]
        examples_filepath = fns[0]
    
    dataset = datasets.load_dataset('json', data_files=[examples_filepath], field='data', keep_in_memory=True, split='train', cache_dir=os.path.join(scratch_dirpath, '.cache'))
    
    tokenizer = torch.load(tokenizer_filepath)
    tokenized_dataset = tokenize_for_qa(tokenizer, dataset)
    tokenized_dataset.set_format('pt', columns=['input_ids', 'attention_mask', 'token_type_ids', 'start_positions', 'end_positions'])
    
    return model,tokenizer,tokenized_dataset;



def evaluate(trigger,start_idx,model,tokenizer,dataset):
    batch_size=10
    
    if trigger is None:
        tokens=None;
    else:
        tokens=text2tokens(tokenizer,trigger);
        tokens=torch.LongTensor(tokens).view(1,-1).cuda();
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size);
    losses=[];
    start_logits=[];
    end_logits=[];
    start_gt=[];
    end_gt=[];
    for data in dataloader:
        with torch.no_grad():
            print(data.keys())
            loss_trigger_i,start_logits_i,end_logits_i=forward(model,data,trigger=tokens,start_idx=start_idx);
            losses.append(loss_trigger_i.data)
            start_logits.append(start_logits_i.data)
            end_logits.append(end_logits_i.data)
            start_gt.append(data['start_positions']);
            end_gt.append(data['end_positions']);
    
    losses=torch.cat(losses,dim=0);
    start_logits=torch.cat(start_logits,dim=0);
    end_logits=torch.cat(end_logits,dim=0);
    start_gt=torch.cat(start_gt,dim=0);
    end_gt=torch.cat(end_gt,dim=0);
    
    return losses,start_logits,end_logits,start_gt,end_gt


# The inferencing approach was adapted from: https://github.com/huggingface/transformers/blob/master/examples/pytorch/question-answering/run_qa.py
def tokenize_for_qa(tokenizer, dataset):
    column_names = dataset.column_names
    question_column_name = "question"
    context_column_name = "context"
    answer_column_name = "answers"
    
    # Padding side determines if we do (question|context) or (context|question).
    pad_on_right = tokenizer.padding_side == "right"
    max_seq_length = min(tokenizer.model_max_length, 384)
    
    if 'mobilebert' in tokenizer.name_or_path:
        max_seq_length = tokenizer.max_model_input_sizes[tokenizer.name_or_path.split('/')[1]]
    
    # Training preprocessing
    def prepare_train_features(examples):
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        
        pad_to_max_length = True
        doc_stride = 128
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if pad_to_max_length else False,
            return_token_type_ids=True)  # certain model types do not have token_type_ids (i.e. Roberta), so ensure they are created
        
        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        # offset_mapping = tokenized_examples.pop("offset_mapping")
        # offset_mapping = copy.deepcopy(tokenized_examples["offset_mapping"])
        
        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []
        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        tokenized_examples["example_id"] = []
        
        for i, offsets in enumerate(tokenized_examples["offset_mapping"]):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)
            
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            
            context_index = 1 if pad_on_right else 0
            
            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = examples[answer_column_name][sample_index]
            # One example can give several spans, this is the index of the example containing this span of text.
            tokenized_examples["example_id"].append(examples["id"][sample_index])
            
            # If no answers are given, set the cls_index as answer.
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])
                
                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                    token_start_index += 1
                
                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                    token_end_index -= 1
                
                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)
            
            # This is for the evaluation side of the processing
            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]
        
        return tokenized_examples
    
    # Create train feature from dataset
    tokenized_dataset = dataset.map(
        prepare_train_features,
        batched=True,
        num_proc=1,
        remove_columns=column_names,
        keep_in_memory=True)
    
    if len(tokenized_dataset) == 0:
        print(
            'Dataset is empty, creating blank tokenized_dataset to ensure correct operation with pytorch data_loader formatting')
        # create blank dataset to allow the 'set_format' command below to generate the right columns
        data_dict = {'input_ids': [],
                     'attention_mask': [],
                     'token_type_ids': [],
                     'start_positions': [],
                     'end_positions': []}
        tokenized_dataset = datasets.Dataset.from_dict(data_dict)
    return tokenized_dataset


#Convert input text (string) to a bunch of tokens (list)
#This is for generating the trigger
def text2tokens(tokenizer,s):
    r=tokenizer(s)
    return r['input_ids'][1:-1];

def mask_logsoftmax(score,mask,dim=1):
    score=score-(1-mask)*1e5;
    return F.log_softmax(score,dim=dim);

def forward(model,data,trigger=None,start_idx=None):
    input_ids = data['input_ids'].cuda()
    attention_mask = data['attention_mask'].cuda()
    token_type_ids = data['token_type_ids'].cuda()
    start_positions = data['start_positions'].cuda()
    end_positions = data['end_positions'].cuda()
    
    if not trigger is None:
        #Shift input and shift output
        assert len(trigger.shape)==2
        if trigger.shape[0]<input_ids.shape[0] and trigger.shape[0]==1:
            trigger=trigger.repeat(input_ids.shape[0],1);
        
        trigger_length=trigger.shape[1];
        input_ids=torch.cat((input_ids[:,:start_idx],trigger,input_ids[:,start_idx:]),dim=1);
        attention_mask=torch.cat((attention_mask[:,:start_idx],trigger*0+1,attention_mask[:,start_idx:]),dim=1);
        token_type_ids=torch.cat((token_type_ids[:,:start_idx],trigger*0,token_type_ids[:,start_idx:]),dim=1);
        
        #Shift start & end positions
        start_positions[start_positions.gt(start_idx)]+=trigger_length;
        end_positions[end_positions.gt(start_idx)]+=trigger_length;
    
    
    model_output_dict = model(input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids,start_positions=start_positions,end_positions=end_positions)
    
    loss=model_output_dict['loss']
    start_logits=model_output_dict['start_logits']
    end_logits=model_output_dict['end_logits'];
    
    start_logits=mask_logsoftmax(start_logits,attention_mask,dim=1);
    end_logits=mask_logsoftmax(end_logits,attention_mask,dim=1);
    
    start_loss=-start_logits.gather(1,start_positions.view(-1,1));
    end_loss=-end_logits.gather(1,end_positions.view(-1,1));
    
    if not trigger is None:
        assert len(trigger.shape)==2
        trigger_length=trigger.shape[1];
        #Shift logits
        start_logits=torch.cat((start_logits[:,:start_idx],start_logits[:,start_idx+trigger_length:]),dim=1);
        end_logits=torch.cat((end_logits[:,:start_idx],end_logits[:,start_idx+trigger_length:]),dim=1);
        
        start_logits=F.log_softmax(start_logits,dim=1);
        end_logits=F.log_softmax(end_logits,dim=1);
    
    
    return (start_loss+end_loss).view(-1)/2,start_logits,end_logits;

def get_we(model):
    we=None;
    try:
        we=model.roberta.embeddings.word_embeddings.weight;
    except:
        pass;
    
    try:
        we=model.electra.embeddings.word_embeddings.weight;
    except:
        pass;
    
    assert not(we is None);
    return we;
    
def id2embed(ids,we):
    e=we[ids.view(-1),:];
    e=e.view(*(list(ids.shape)+[-1]));
    return e;

def forward_embed(model,data,trigger=None,start_idx=None):
    trigger_embed=trigger;
    input_ids = data['input_ids'].cuda()
    attention_mask = data['attention_mask'].cuda()
    token_type_ids = data['token_type_ids'].cuda()
    start_positions = data['start_positions'].cuda()
    end_positions = data['end_positions'].cuda()
    
    we=get_we(model)
    
    if not trigger_embed is None:
        #Shift input and shift output
        assert len(trigger_embed.shape)==3
        trigger_length=trigger_embed.shape[1];
        if trigger_embed.shape[0]<input_ids.shape[0] and trigger_embed.shape[0]==1:
            trigger_embed=trigger_embed.repeat(input_ids.shape[0],1,1);
        
        input_embeds=id2embed(input_ids,we);
        
        input_embeds=torch.cat((input_embeds[:,:start_idx,:],trigger_embed,input_embeds[:,start_idx:,:]),dim=1);
        attention_mask=torch.cat((attention_mask[:,:start_idx],trigger_embed[:,:,0].long()*0+1,attention_mask[:,start_idx:]),dim=1);
        token_type_ids=torch.cat((token_type_ids[:,:start_idx],trigger_embed[:,:,0].long()*0,token_type_ids[:,start_idx:]),dim=1);
        
        #Shift start & end positions
        start_positions[start_positions.gt(start_idx)]+=trigger_length;
        end_positions[end_positions.gt(start_idx)]+=trigger_length;
    
    
    model_output_dict = model(None,attention_mask=attention_mask,token_type_ids=token_type_ids,start_positions=start_positions,end_positions=end_positions,inputs_embeds=input_embeds)
    
    loss=model_output_dict['loss']
    start_logits=model_output_dict['start_logits']
    end_logits=model_output_dict['end_logits'];
    
    start_logits=mask_logsoftmax(start_logits,attention_mask,dim=1);
    end_logits=mask_logsoftmax(end_logits,attention_mask,dim=1);
    
    start_loss=-start_logits.gather(1,start_positions.view(-1,1));
    end_loss=-end_logits.gather(1,end_positions.view(-1,1));
    
    if not trigger_embed is None:
        assert len(trigger_embed.shape)==3
        trigger_length=trigger_embed.shape[1];
        #Shift logits
        start_logits=torch.cat((start_logits[:,:start_idx],start_logits[:,start_idx+trigger_length:]),dim=1);
        end_logits=torch.cat((end_logits[:,:start_idx],end_logits[:,start_idx+trigger_length:]),dim=1);
        
        start_logits=F.log_softmax(start_logits,dim=1);
        end_logits=F.log_softmax(end_logits,dim=1);
    
    
    return (start_loss+end_loss).view(-1)/2,start_logits,end_logits;

if __name__ == "__main__":
    
    examples_filepath='data/round8-train-dataset/models/id-00000000/example_data/clean-example-data.json'
    scratch_dirpath='./scratch'
    tokenizer_filepath='data/round8-train-dataset/tokenizers/tokenizer-roberta-base.pt';
    model_filepath='data/round8-train-dataset/models/id-00000000/model.pt'
    
    
    dataset = datasets.load_dataset('json', data_files=[examples_filepath], field='data', keep_in_memory=True, split='train', cache_dir=os.path.join(scratch_dirpath, '.cache'))
    
    
    tokenizer = torch.load(tokenizer_filepath)
    tokenized_dataset = tokenize_for_qa(tokenizer, dataset)
    tokenized_dataset.set_format('pt', columns=['input_ids', 'attention_mask', 'token_type_ids', 'start_positions', 'end_positions'])
    dataloader = torch.utils.data.DataLoader(tokenized_dataset, batch_size=5)
    
    
    model = torch.load(model_filepath).cuda()
    model.eval()
    
    we=get_we(model);
    
    trigger=text2tokens(tokenizer,'Typical values for common groups of humans follow');
    trigger=torch.LongTensor(trigger).view(1,-1).cuda();
    trigger=id2embed(trigger,we);
    
    
    fake_trigger=text2tokens(tokenizer,'Typical ');
    fake_trigger=torch.LongTensor(fake_trigger).view(1,-1).cuda();
    fake_trigger=id2embed(fake_trigger,we);
    
    losses=[];
    for data in dataloader:
        loss,_,_=forward(model,data);
        losses.append(float(loss.mean()))
    
    print('clean %f'%(sum(losses)/len(losses)))
    
    #Not triggering
    losses=[];
    for data in dataloader:
        loss,_,_=forward_embed(model,data,trigger=trigger,start_idx=1);
        losses.append(float(loss.mean()))
    
    print('real trig %f'%(sum(losses)/len(losses)))
    
    losses=[];
    for data in dataloader:
        loss,_,_=forward_embed(model,data,trigger=fake_trigger,start_idx=1);
        losses.append(float(loss.mean()))
    
    print('fake trig %f'%(sum(losses)/len(losses)))
