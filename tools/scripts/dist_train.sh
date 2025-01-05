#!/usr/bin/env bash

set -x
NGPUS=$1 #GPU数量
PY_ARGS=${@:2} #除GPU数量外的其他参数

while true #随机选取一个端口
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done
echo $PORT

python -m torch.distributed.launch --nproc_per_node=${NGPUS} --rdzv_endpoint=localhost:${PORT} train.py --launcher pytorch ${PY_ARGS}

