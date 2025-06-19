#!/bin/bash
set -euxo pipefail

timestamp_suffix=$(date '+%Y-%m-%d_%H:%M:%S')

LOG_ROOT_DIR=""
TENSORBOARD_DIR="${LOG_ROOT_DIR}/rl_"${timestamp_suffix}

DATA_DIR="distill_data"
STUDENT_MODEL="path_of_initial_student_model"
TEACHER_MODEL="path_of_teacher_model"
MODEL_SAVE_DIR="save_dir"

WORKING_DIR="working_dir"


ray job submit --no-wait --address="http://xxx.xxx.xxx.xxx:port" \
    --runtime-env-json='{"working_dir": "'${WORKING_DIR}'", "env_vars": {"GLOO_SOCKET_IFNAME": "eth0", "TENSORBOARD_DIR": "'${TENSORBOARD_DIR}'"}}' \
    -- python3 -m verl.trainer.main_ppo \
    data.train_files=${DATA_DIR}/train.parquet \
    data.val_files=${DATA_DIR}/test.parquet \
    data.prompt_key=prompt \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    data.train_batch_size=256 \
    data.filter_overlong_prompts=True \
    data.shuffle=True \
    +data.seed=42 \
    data.truncation='error' \
    actor_rollout_ref.model.path=${STUDENT_MODEL} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.2 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=5 \
    actor_rollout_ref.actor.optim.min_lr_ratio=0.1 \
    actor_rollout_ref.actor.optim.warmup_style=cosine \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1.2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.n=16 \
    distill_knowledge.enable=True \
    distill_knowledge.temperature=1.0 \
    distill_knowledge.log_prob_micro_batch_size_per_gpu=32 \
    distill_knowledge.teacher.path=${TEACHER_MODEL} \
    distill_knowledge.teacher.use_remove_padding=True \
    algorithm.adv_estimator=revert_prob \
    algorithm.use_kl_in_reward=False \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name='on_policy_kd' \
    trainer.experiment_name='qwen3_8b_kd' \
    trainer.default_local_dir=${MODEL_SAVE_DIR} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=2 \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.total_epochs=1
