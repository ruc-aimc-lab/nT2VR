#!/bin/sh
source activate py37


# *************************萌萌哒******************************
# 并行参数
Nproc=1 # 最大同s时运行数目
devices=(1)
# 实验超参
rootpath="/data1/wzy/neg_data"
trainCollection="msrvtt10ktrain"
valCollection="msrvtt10kval"
val_set='no'
testCollection="msrvtt10ktest"
config='CLIP.CLIPEnd2EndNegnomask'
pretrained_file_path=None
batch_size=64
overwrite=0
task3suffix="negation"
random_seeds=(2)  # 初始化随机数种子
#parm_adjust_configs=(1)


parm_adjust_configs=(     "1_0.001_0.1_0.6_100_0.1_0.3_0.001"  )

echo ${parm_adjust_configs[*]}

model_prefix_="runs_"
result_file="result_log/result_${model_prefix_}_${config}_${task3suffix}.txt"

cd ..

bash retrieval_task2.sh --rootpath $rootpath --trainCollection $trainCollection  --valCollection $valCollection \
--val_set $val_set --testCollection $testCollection \
--config $config  --batch_size $batch_size  --overwrite $overwrite   \
--devices "${devices[*]}" --Nproc $Nproc --parm_adjust_configs "${parm_adjust_configs[*]}" \
--model_prefix_ $model_prefix_ --result_file $result_file --random_seeds "${random_seeds[*]}"  --task3_caption $task3suffix
