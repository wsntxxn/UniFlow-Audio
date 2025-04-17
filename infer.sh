#./sh/sft_jiahao_multinode.sh 内容

#!/bin/bash

# 解析命令行参数
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --name) name="$2"; shift ;;  # 解析 --name 参数
        *) echo "Unknown parameter: $1" ; exit 1 ;;
    esac
    shift
done

# 检查 name 是否提供
if [ -z "$name" ]; then
    echo "Error: --name 参数未提供"
    exit 1
fi
echo "Current time: $(TZ='Asia/Shanghai' date '+%Y-%m-%d %H:%M:%S')"
echo "Running script name: $(basename "$0")" 
echo "start training"

> logs/${name}.log 2>&1


export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="/cpfs_shared/jiahao.mei/hf_home"


echo "Current time: $(TZ='Asia/Shanghai' date '+%Y-%m-%d %H:%M:%S')" >> logs/${name}.log 2>&1
echo "Running script name: $(basename "$0")" >> logs/${name}.log 2>&1

source  /cpfs_shared/jiahao.mei/miniconda3/bin/activate /cpfs_shared/jiahao.mei/miniconda3/envs/x2audio/. >> logs/${name}.log 2>&1
which python >> logs/${name}.log 2>&1
wandb login "88b3c37b4496e45f2e51386b7d72ffe2699e5339" >> logs/${name}.log 2>&1
pwd >> logs/${name}.log 2>&1

cat ~/.bashrc >> logs/${name}.log 2>&1


bash ./bash_scripts/${name}.sh  >> logs/${name}.log 2>&1

echo "Done" 