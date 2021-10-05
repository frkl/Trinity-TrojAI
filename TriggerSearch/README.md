## Introduction

This folder contains code & data for a surrogate-based Trojan reverse engineering method for [TrojAI challenge](https://pages.nist.gov/trojai/) round 8 Question Answering, which achieved CE 0.5749 and ROC-AUC 0.7593. 

Surrogate-based Trojan reverse engineering addresses the challenge of reverse engineering Trojan triggers within a limited number of inferences to the original transformer. A fast surrogate model which takes a trigger phrase as input and predict its label flipping loss is learned to mimic the original transformer. Triggers that achieve low loss on the surrogate model are iteratively selected to query the original transformer and the query result is used for updating the surrogate model. 

To improve the accuracy of the surrogate model under few updates, we use meta-learning to learn a trigger embedding function which, under linear regression, mimics the original transformer well.

## Prerequisite

Extract TrojAI round 8 models into `data/round8-train-dataset/models`. METADATA.csv at `data/round8-train-dataset/METADATA.csv`.

Follow https://github.com/usnistgov/trojai-example to setup your TrojAI round 8 conda virtual environment.

## Common Tasks

### Extract meta dataset for surrogate learning

`python data_meta_r8_sel2.py` extracts "traces" of 50000 points in `meta-r8-sel2/`, one file for each model. For each model, we sample 50000 random phrases using the model's dictionary, and record the model's response on a fixed set of 12 questions in `selected_data.json` when each phrase is inserted in 1 of 3 positions of the question, resulting in a 15x3x5000 tensor per model. 

This process takes ~4 weeks on a single GPU. To shorten the wait time, edit the model id range in `python data_meta_r8_sel2.py` manually and run different models on different GPUs.

A pre-extracted `meta-r8-sel2/` dataset is hosted at https://www.dropbox.com/s/650taoo7l3gqsc9/meta-r8-sel2.zip?dl=0 .

### Meta-learning the surrogate model

`python meta-v4-log2.py` loads the extracted traces in `meta-r8-sel2/`, and train a linear surrogate model using meta-learning. It learns a MLP that convert phrases of 8 word embeddings into a single 512-dim embedding, which under linear regression predicts losses of unseen phrases well given losses on seen phrases. The architecture of the surrogate can be found in `arch/maml_surrogate_linear_gp_multi.py`

`linear_512_200v2.pt` is 200 epochs into the most recent run using the traces in `meta-r8-sel2/`.

### Reverse engineering Trojans using the surrogate model

`python extract_fvs_bo7.py` uses the surrogate models to suggest triggers to try for reverse engineering. Parameters are hard coded in the `def extract_bo_features(...)` function:

* `checkpoint` specifies the surrogate model to use. 

* `maxl` specifies the maximum phrase length of the surrogate model (My surrogates are all designed for 8 word phrases). 

* `actual_l` specifies the maximum phrase length to search. 

* `niter` specifies the number of optimization rounds. 

Triggers for the first 20 rounds are randomly generated. And then the surrogate suggests `actual_l` triggers to try each round for `niter` rounds. The pre-selected examples used for feature extraction is `selected_data_36.json` for bo7. Examples of extracted features using `linear_512_200v2.pt` is `data_r8_surrogate7_linear_512_200_v2.pt`.

### Trojan detection using reverse engineered losses

After feature extraction, use `python crossval_hyper.py --data data_r8_surrogate7_linear_512_800.pt --arch arch.mlp_known_v5` to train a Trojan classifier using crossval. 

`session_0000152` contains a pretrained Trojan detector using the reverse engineered module in `extract_fvs_bo7.py`.  

`real_trojan_detector.def` was the singularity container definition file for our latest submission, which achieved CE 0.5749 and ROC-AUC 0.7593 on the round 8 leaderboard. 

`build.sh` has a testing script for building and testing the singularity container.

### References

If you use our code as part of published research, please cite:

```
@misc{Lin2021,
author = {Xiao Lin and Michael Cogswell and Meng Ye and Ajay Divakaran and Susmit Jha and Yi Yao},
title = {Surrogate-based Reverse Engineering for Trojan Detection},
year = {2021},
publisher = {GitHub},
journal = {GitHub repository},
howpublished = {\url{https://github.com/frkl/Trinity-TrojAI/tree/trojai-r8/TriggerSearch}},
}```
