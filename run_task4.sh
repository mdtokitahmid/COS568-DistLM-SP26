#!/bin/bash

export TASK_NAME=RTE
export GLUE_DIR=$HOME/glue_data
export GLOO_SOCKET_IFNAME=enp130s0f0

RANK=$1
mkdir -p ~/COS568-DistLM-SP26/profiler_output

# echo "=== Running Task 4 - 2a ==="
# python3 task4/run_glue_2a.py --model_type bert --model_name_or_path bert-base-cased --task_name $TASK_NAME --do_train --do_eval --data_dir $GLUE_DIR/$TASK_NAME --max_seq_length 128 --per_device_train_batch_size 16 --learning_rate 2e-5 --num_train_epochs 1 --output_dir /tmp/$TASK_NAME/ --overwrite_output_dir --master_ip 10.10.1.2 --master_port 12345 --world_size 4 --local_rank $RANK

echo "=== Running Task 4 - 2b ==="
python3 task4/run_glue_2b.py --model_type bert --model_name_or_path bert-base-cased --task_name $TASK_NAME --do_train --do_eval --data_dir $GLUE_DIR/$TASK_NAME --max_seq_length 128 --per_device_train_batch_size 16 --learning_rate 2e-5 --num_train_epochs 1 --output_dir /tmp/$TASK_NAME/ --overwrite_output_dir --master_ip 10.10.1.2 --master_port 12346 --world_size 4 --local_rank $RANK

# echo "=== Running Task 4 - 3 ==="
# python3 task4/run_glue_3.py --model_type bert --model_name_or_path bert-base-cased --task_name $TASK_NAME --do_train --do_eval --data_dir $GLUE_DIR/$TASK_NAME --max_seq_length 128 --per_device_train_batch_size 16 --learning_rate 2e-5 --num_train_epochs 1 --output_dir /tmp/$TASK_NAME/ --overwrite_output_dir --master_ip 10.10.1.2 --master_port 12347 --world_size 4 --local_rank $RANK

echo "=== Done ==="
