# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import os

import datasets
import numpy as np
import torch
import transformers
import json
import numpy
import time
import sklearn.metrics
import importlib
import torch.nn.functional as F
import math
import pickle

import warnings

import utils_qa

warnings.filterwarnings("ignore")


def example_trojan_detector(model_filepath, tokenizer_filepath, result_filepath, scratch_dirpath, examples_dirpath):
    try:
        checkpoint=torch.load('session_0000152/model.pt');
    except:
        checkpoint=torch.load('/session_0000152/model.pt');
    
    
    import extract_fvs_bo7 as fvs
    fvs,tokens = fvs.extract_fvs(model_filepath, tokenizer_filepath, scratch_dirpath, examples_dirpath) 
    
    
    fvs=torch.log(fvs.clamp(min=1e-20,max=1e20));
    fvs=fvs.view(540,3*36);
    fvs,_=torch.cummin(fvs,dim=0);
    #x=x.mean(dim=-1);
    #x=x.view(-1);
    #x,_=x.sort(dim=0)
    #ind=[0,59,251];
    fvs=fvs[[0,119,539],:].clone() #50
    fvs=fvs.view(1,-1);
    
    fvs = {
        'ws_surrogate':fvs,'tokens_surrogate':[tokens],
    }
    
    s=[];
    
    for i in range(len(checkpoint)):
        params_=checkpoint[i]['params'];
        arch_=importlib.import_module(params_.arch);
        net=arch_.new(params_);
        
        net.load_state_dict(checkpoint[i]['net']);
        net=net.cuda();
        net.eval();
        
        s_i=net.logp(fvs).data.cpu()*math.exp(-checkpoint[i]['T']);
        s.append(float(s_i))
    
    s=sum(s)/len(s);
    s=torch.sigmoid(torch.Tensor([s]));
    trojan_probability=float(s);
    
    
    print('Trojan Probability: {}'.format(trojan_probability))
    
    with open(result_filepath, 'w') as fh:
        fh.write("{}".format(trojan_probability))
    
    return trojan_probability;


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Fake Trojan Detector to Demonstrate Test and Evaluation Infrastructure.')
    parser.add_argument('--model_filepath', type=str, help='File path to the pytorch model file to be evaluated.', default='./model/model.pt')
    parser.add_argument('--tokenizer_filepath', type=str, help='File path to the pytorch model (.pt) file containing the correct tokenizer to be used with the model_filepath.', default='./tokenizers/google-electra-small-discriminator.pt')
    parser.add_argument('--result_filepath', type=str, help='File path to the file where output result should be written. After execution this file should contain a single line with a single floating point trojan probability.', default='./output.txt')
    parser.add_argument('--scratch_dirpath', type=str, help='File path to the folder where scratch disk space exists. This folder will be empty at execution start and will be deleted at completion of execution.', default='./scratch')
    parser.add_argument('--examples_dirpath', type=str, help='File path to the directory containing json file(s) that contains the examples which might be useful for determining whether a model is poisoned.', default='./model/example_data')

    args = parser.parse_args()

    example_trojan_detector(args.model_filepath, args.tokenizer_filepath, args.result_filepath, args.scratch_dirpath, args.examples_dirpath)