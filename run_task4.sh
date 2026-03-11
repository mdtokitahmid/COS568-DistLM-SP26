#!/bin/bash

export TASK_NAME=RTE
export GLUE_DIR=$HOME/glue_data

MASTER_IP=10.10.1.2
MASTER_PORT=12345
WORLD_SIZE=4
RANK=$1  # pass rank as argument: ./run_task4.sh 0

mkdir -p ~/COS568-DistLM-SP26/profiler_output

BASE_ARGS="--model_type bert --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME --do_train --do_eval \
  --data_dir $GLUE_DIR/$TASK_NAME --max_seq_length 128 \
  --per_device_train_batch_size 16 --learning_rate 2e-5 \
  --num_train_epochs 1 --output_dir /tmp/$TASK_NAME/ \
  --overwrite_output_dir \
  --master_ip $MASTER_IP --master_port $MASTER_PORT \
  --world_size $WORLD_SIZE --local_rank $RANK"

echo "=== Running Task 4 - 2a (gather/scatter) ==="
python3 task4/run_glue_2a.py $BASE_ARGS

echo "=== Running Task 4 - 2b (all_reduce) ==="
python3 task4/run_glue_2b.py $BASE_ARGS

echo "=== Running Task 4 - 3 (DDP) ==="
python3 task4/run_glue_3.py $BASE_ARGS

echo "=== Done ==="
