#!/bin/sh
source activate py37


# *************************萌萌哒******************************
# 并行参数
Nproc=1 # 最大同时运行数目
devices=(2)
rootpath="/data1/wzy/neg_data"
val_set='no'
config='CLIP.CLIPEnd2End_adjust'
# msrvtt 1k split
#trainCollection="msrvtt1kAtrain"
#valCollection="msrvtt1kAval"
#testCollection="msrvtt1kAtest"

# msrvtt 7k split
#trainCollection="msrvtt10ktrain"
#valCollection="msrvtt10kval"
#testCollection="msrvtt10kest"

# vatex
trainCollection="vatex_train"
valCollection="vatex_val1k5"
testCollection="vatex_test1k5"

pretrained_file_path=None
batch_size=64
overwrite=0
task3suffix="no_task3_caption"
random_seeds=(2)  # 初始化随机数种子
#parm_adjust_configs=(1)


parm_adjust_configs=(     "1"  )

echo ${parm_adjust_configs[*]}

model_prefix_="runs_"
result_file="result_log/result_${model_prefix_}_${config}_${task3suffix}.txt"

cd ..

bash retrieval_task2.sh --rootpath $rootpath --trainCollection $trainCollection  --valCollection $valCollection \
--val_set $val_set --testCollection $testCollection \
--config $config  --batch_size $batch_size  --overwrite $overwrite   \
--devices "${devices[*]}" --Nproc $Nproc --parm_adjust_configs "${parm_adjust_configs[*]}" \
--model_prefix_ $model_prefix_ --result_file $result_file --random_seeds "${random_seeds[*]}"  --task3_caption $task3suffix
