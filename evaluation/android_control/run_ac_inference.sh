set -x

DATASET=${1}
CHECKPOINT=${2}
echo "DATASET: ${DATASET}"
echo "CHECKPOINT: ${CHECKPOINT}"

# 检查数据集名称是否正确
options=("ac_high" "ac_low") 
if [[ " ${options[@]} " =~ " ${DATASET} " ]]; then
    echo "输入 '$DATASET' 是有效选项"
else
    echo "错误: '$DATASET' 不是有效选项"
    echo "有效选项是: ${options[@]}"
    exit 1
fi

PATCH=24

TEST_SET_NAME=${DATASET}
echo "测试集key： ${TEST_SET_NAME}"

MASTER_PORT=${MASTER_PORT:-63604}
PORT=${PORT:-63604}
PARTITION=${PARTITION:-"INTERN2"}

export MASTER_PORT=${MASTER_PORT}
export PORT=${PORT}
echo "GPUS: ${GPUS}"

NODES=$((GPUS / GPUS_PER_NODE))
GPUS_PER_NODE=${GPUS_PER_NODE:-${GPUS}}
QUOTA_TYPE=${QUOTA_TYPE:-"reserved"}
SRUN_ARGS=${SRUN_ARGS:-""}


your/conda/path/torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    internvl2_inference.py --checkpoint ${CHECKPOINT_DIR} --datasets ${DATASET} \
      --dynamic \
      --max-num ${PATCH} \
      --ds_name_list ${TEST_SET_NAME} \
      --out-dir ${OUT_DIR};"

