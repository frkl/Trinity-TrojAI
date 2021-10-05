
#test container

python real_trojan_detector.py \
--model_filepath=./data/round8-train-dataset/models/id-00000000/model.pt \
--tokenizer_filepath=./data/round8-train-dataset/tokenizers/tokenizer-deepset-roberta-base-squad2.pt \
--result_filepath=./output.txt \
--scratch_dirpath=./scratch/ \
--examples_dirpath=./data/round8-train-dataset/models/id-00000000/example_data/


sudo singularity build real_trojan_detector.simg real_trojan_detector.def