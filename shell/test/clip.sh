#!/bin/sh
source activate py37


# *************************萌萌哒******************************
# 并行参数
Nproc=1  # 最大同时运行数目
devices=(0)
# 实验超参
rootpath="/data1/wzy/neg_data"

# msrvtt 7k split
#testCollection="msrvtt10ktest"

# msrvtt 1k split
#testCollection="msrvtt1kAtest"

# vatex
testCollection="vatex_test1k5"

config='CLIP.CLIP'
batch_size=128

overwrite=1

task3suffix="no_task3_caption"
random_seeds=(2)  # 初始化随机数种子
parm_adjust_configs=(  "1" )

model_prefix_="runs_"
result_file="result_log/result_${model_prefix_}_${config}.txt"

cd ..
bash retrieval_task2.sh --rootpath $rootpath  --testCollection $testCollection \
--config $config  --batch_size $batch_size  --overwrite $overwrite \
--devices "${devices[*]}" --Nproc $Nproc --parm_adjust_configs "${parm_adjust_configs[*]}" \
--model_prefix_ $model_prefix_ --result_file $result_file --random_seeds "${random_seeds[*]}"  --task3_caption $task3suffix \
--model_path 'None'
