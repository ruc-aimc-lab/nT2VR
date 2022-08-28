#!/bin/sh
source activate py37
root_path="/data1/wzy/neg_data/"
#path of models used in negation scope detection model
cache_dir="/data1/wzy/negbert"
# python -m spacy download en_core_web_sm

## training data
#caption_file="/home/wzy/VisualSearch/msrvtt10ktrain/TextData/msrvtt10ktrain.caption.txt"
#dataset="msrvtt10ktrain"
#python generate_negated.py --root_path $root_path --caption_file $caption_file --dataset $dataset

## negated test data
caption_file="/data1/wzy/VisualSearch/msrvtt1kAtest/TextData/msrvtt1k_all.caption.txt"
dataset="msrvtt1kAtest"
python generate_negated.py --root_path $root_path --caption_file $caption_file --dataset $dataset \
--test True --cache_dir $cache_dir
#composed test data
python generate_composed.py --root_path $root_path --caption_file $caption_file --dataset $dataset