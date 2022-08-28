#!/bin/sh
source activate py37


# *************************萌萌哒******************************
# 并行参数
Nproc=1  # 最大同时运行数目
devices=1
# 实验超参
rootpath="/data1/wzy/neg_data"

# msrvtt 7k split
testCollection="msrvtt10ktest"

# msrvtt 1k split
testCollection="msrvtt1kAtest"

## vatex
#testCollection="vatex_test1k5"

config='CLIP.CLIP'
batch_size=128

overwrite=1

task3suffix="no_task3_caption"
random_seeds=(2)  # 初始化随机数种子
parm_adjust_configs=(  "1" )

model_prefix_="runs_"
result_file="result_log/result_${model_prefix_}_bool_${config}.txt"
workers=16
cd ..
cd ..
#             ##bool predict
## predict negated, original
original_cap=$testCollection.captionsubset_neginfo.txt
negated_cap=$testCollection.negated_neginfo.txt
query_sets=$negated_cap,$original_cap
config="bool_${config}"
sim_name=$config
model_path=None
#
  python predictorsub.py $testCollection $model_path $sim_name \
      --query_sets $query_sets  --config_name $config \
      --rootpath $rootpath  --overwrite $overwrite --device $devices \
      --batch_size $batch_size --predict_result_file $result_file --num_workers $workers
  python predict_compute_delta.py $testCollection  $sim_name \
           --config_name $config --model_path $model_path --original_cap $original_cap --negated_cap $negated_cap \
          --rootpath $rootpath  --predict_result_file $result_file

#                 ##      predict composed
   query_sets=$testCollection.composed.txt
  python predictorsub.py $testCollection $model_path $sim_name \
      --query_sets $query_sets  --config_name $config \
      --rootpath $rootpath  --overwrite $overwrite --device $devices \
      --batch_size $batch_size --predict_result_file $result_file --num_workers $workers
