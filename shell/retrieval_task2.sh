#!/bin/sh

<<COMMENT
    这个脚本是把训练，生成测试文件，生成 avs检索 结果三部分结合起来，自动进行。
    注意：
        1. 更改模型在 config 文件中更改。
        2. random_seed 是传入 do_train.py 文件的一个参数，如果没有使用可以修改或者删除。
        3. result_file 是存储结果的文件名。
        3. parm_adjust_configs 是传入 config 文件的一个参数，如果没有使用可以修改或者删除。
COMMENT


# *************************萌萌哒******************************
# 并行参数
Nproc=1    # 可同时运行的最大作业数
devices=()
# shellcheck disable=SC2113
function PushQue {    # 将PID压入队列
	Que="$Que $1"
	Nrun=$(($Nrun+1))
}
function GenQue {     # 更新队列
	OldQue=$Que
	Que=""; Nrun=0
	for PID in $OldQue; do
		if [[ -d /proc/$PID ]]; then
			PushQue $PID
		fi
	done
}
function ChkQue {     # 检查队列
	OldQue=$Que
	for PID in $OldQue; do
		if [[ ! -d /proc/$PID ]] ; then
			GenQue; break
		fi
	done
}
function paralle {
    PID=$!
	PushQue $PID
	while [[ $Nrun -ge ${Nproc} ]]; do
		ChkQue
		sleep 1
	done
}


# *************************萌萌哒******************************
path_shell=`pwd`
cd ..
path_w2vvpp=`pwd`

rootpath="/data/wzy/VisualSearch"
trainCollection=""
valCollection=""
val_set=''  # setA
testCollection='7'
txt_feature_task2='no'
trainCollection2='None'
config=''
batch_size=256
workers=16
overwrite=0
model_path="notNone"
random_seeds=(2)  # 初始化随机数种子
pretrained_file_path='None'

#parm_adjust_configs=()
num_epoch=10
model_prefix_="runs_"

# 读取输入的参数
GETOPT_ARGS=$(getopt -o l: -al rootpath:,trainCollection:,valCollection:,val_set:,testCollection:,config:,batch_size:,overwrite:,devices:,Nproc:,random_seeds:,task3_caption:,parm_adjust_configs:,\
model_prefix_:,result_file:,pretrained_file_path:,model_path:,num_epoch: -- "$@")  # , 后一定不要有空格
eval set -- "$GETOPT_ARGS"
result_file="$path_w2vvpp/result_log/result_${model_prefix_}_${config}.txt"

#获取参数
while [ -n "$1" ]
do
        case "$1" in
                --rootpath) rootpath=$2; shift 2;;
                --trainCollection) trainCollection=$2; shift 2;;
                --valCollection) valCollection=$2; shift 2;;
                --val_set) val_set=$2; shift 2;;
                --testCollection) testCollection=$2; shift 2;;
                --config) config=$2; shift 2;;
                --batch_size) batch_size=$(($2)); shift 2;;
                --overwrite) overwrite=$2; shift 2;;
                --devices) devices_temp=$2; shift 2;;
                --Nproc) Nproc=$(($2)); shift 2;;
                --parm_adjust_configs) parm_adjust_configs=$2; shift 2;;
                --random_seeds) random_seeds=$2; shift 2;;
                --pretrained_file_path) pretrained_file_path=$2; shift 2;;
                --model_prefix_) model_prefix_=$2; shift 2;;
                --result_file) result_file=$2; shift 2;;
                --task3_caption) task3_caption=$2; shift 2;;
              --model_path) model_path=$2; shift 2;;
                --num_epoch) num_epoch=$(($2)); shift 2;;
                --) break ;;
                *) echo $1,$2; break ;;
        esac
done

echo "result_file：$result_file,num_epoch：${num_epoch}, devices: ${devices[*]} , config: $config , parm_adjust_configs: ${parm_adjust_configs},
 Nproc: ${Nproc} testCollection:$testCollection"

for each in ${devices_temp[*]}
do
    devices[${#devices[@]}]=$each
done
#exit 0


# ****************************************
# 训练
# shellcheck disable=SC2039

for random_seed in ${random_seeds[*]}
do

    for parm_adjust_config in ${parm_adjust_configs[*]}
    do

        model_prefix="${model_prefix_}${parm_adjust_config}_seed_${random_seed}"



        device=${devices[device_index]}
        let device_index="($device_index + 1) % ${#devices[*]}"

      if [ $model_path == 'notNone' ]; then
        if [[ $Nproc -gt 1 ]]; then
            echo "$trainCollection $valCollection --rootpath $rootpath --config $config --val_set $val_set --model_prefix $model_prefix --device $device "

            python do_trainer.py $trainCollection $valCollection \
                --rootpath $rootpath --config_name $config --val_set $val_set --model_prefix $model_prefix \
                --batch_size $batch_size --workers $workers --device $device --overwrite $overwrite \
                --parm_adjust_config $parm_adjust_config --num_epochs $num_epoch \
                --random_seed $random_seed  --task3_caption $task3_caption  --pretrained_file_path $pretrained_file_path\
                & paralle
        else
echo "$trainCollection $valCollection --rootpath $rootpath --config $config --val_set $val_set --model_prefix $model_prefix --device $device "
            python do_trainer.py $trainCollection $valCollection \
                --rootpath $rootpath --config_name $config --val_set $val_set --model_prefix $model_prefix \
                --batch_size $batch_size --workers $workers --device $device --overwrite $overwrite \
                --parm_adjust_config $parm_adjust_config --num_epochs $num_epoch \
                --random_seed $random_seed  --task3_caption $task3_caption  --pretrained_file_path $pretrained_file_path
        fi
        fi



    overwrite=1

       model_path0=$rootpath/$trainCollection/w2vvpp_train/$valCollection/$config/$model_prefix
    model_names=('model_best.pth.tar')

      for model_name in ${model_names[*]}
        do
             if [ $model_path == 'notNone' ] ;then
            model_path=$model_path0/$model_name
            sim_name=$config/$model_prefix/$model_name

             else

               sim_name=$config
            fi



             original_cap=$testCollection.caption.txt
             negated_cap=$testCollection.negated.txt
             query_sets=$original_cap,$negated_cap
            #query_sets=simple_query.txt
            python predictor.py $testCollection $model_path $sim_name \
                --query_sets $query_sets --config_name $config \
                --rootpath $rootpath  --overwrite $overwrite --device $device \
                --batch_size $batch_size --predict_result_file $result_file --num_workers $workers --task3_caption $task3_caption

            python predict_compute_delta.py $testCollection  $sim_name \
                 --config_name $config --model_path $model_path --original_cap $original_cap --negated_cap $negated_cap \
                --rootpath $rootpath  --predict_result_file $result_file

            query_sets=$testCollection.composed.txt

            python predictor.py $testCollection $model_path $sim_name \
                --query_sets $query_sets --config_name $config \
                --rootpath $rootpath  --overwrite $overwrite --device $device \
                --batch_size $batch_size --predict_result_file $result_file --num_workers $workers --task3_caption $task3_caption \
                --adhoc True

  model_path='notNone'
  done

    done
done
wait
