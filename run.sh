#!/bin/bash 
# The interpreter used to execute the script
# Lines beginning with #SBATCH specify your computing resources and other logistics about how to run your job.

# Access computing resources allocated to the MLRE course account, the section may be 006 or 007 depending on the student.
#SBATCH --account=eecs498f25s006_class

# Specify the maximum runtime (in Hours:Minutes:Seconds). If your job hits that runtime, it will be terminated.
#SBATCH --time=7:45:00

# Specify whether to use a GPU (partition=gpu) or CPU (partition=standard). If you use the GPU partition, only request one GPU (gpus=1).
# Note that if you use the standard partition, you may have to remove the gpu configuration.
#SBATCH --partition=gpu
#SBATCH --gpus=1

# There are also more specific settings for configuring your CPU/GPU. You can reference the documentation for more information.
# Specify the amount of memory you need. If your job exceeds this memory limit, it will be terminated.
#SBATCH --mem=64g

# Name this job and the output file
#SBATCH --job-name=rlvlmf_llama_test_1
#SBATCH --output=out_files/llama_test_1.out

# Receive an email when your job starts and ends
#SBATCH --mail-user=pajehan@umich.edu
#SBATCH --mail-type=BEGIN,END

# Load Python and any other desired modules (we are using a conda env)
# module purge
# module load python/3.9

# Specify the script you want to run

# ### ClothFold
# python train_PEBBLE.py \
#     env=softgym_ClothFoldDiagonal \
#     seed=0 \
#     vlm_label=1 \
#     vlm=gpt4v_two_image \
#     exp_name=test \
#     reward=learn_from_preference \
#     image_reward=1 \
#     num_train_steps=15000 \
#     agent.params.actor_lr=0.0005 agent.params.critic_lr=0.0005 gradient_update=1 activation=tanh num_unsup_steps=250 \
#     num_interact=1000 max_feedback=500 \
#     reward_batch=50 reward_update=25 \
#     resnet=1 \
#     feed_type=0 teacher_beta=-1 teacher_gamma=1 teacher_eps_mistake=0 \
#     teacher_eps_skip=0 teacher_eps_equal=0 segment=1 num_seed_steps=250 \
#     eval_frequency=250 num_eval_episodes=1 \
#     cached_label_path=data/cached_labels/ClothFold/seed_0/


### soccer
python train_PEBBLE.py \
    env=metaworld_soccer-v2 \
    seed=0 \
    exp_name=reproduce \
    reward=learn_from_score \
    vlm_label=1 \
    vlm=gemini_free_form \
    image_reward=1 \
    reward_batch=40 \
    segment=1 \
    teacher_eps_mistake=0 \
    reward_update=5 \
    num_interact=4000 \
    max_feedback=20000 \
    reward_lr=1e-4 \
    agent.params.actor_lr=0.0003 agent.params.critic_lr=0.0003 gradient_update=1 activation=tanh num_unsup_steps=9000 \
    num_train_steps=400000 agent.params.batch_size=512 double_q_critic.params.hidden_dim=256 double_q_critic.params.hidden_depth=3 \
    diag_gaussian_actor.params.hidden_dim=256 diag_gaussian_actor.params.hidden_depth=3  \
    feed_type=0 teacher_beta=-1 teacher_gamma=1  teacher_eps_skip=0 teacher_eps_equal=0 \
    num_eval_episodes=1 #\
    #cached_label_path=data/cached_labels/Soccer/seed_1/


# ### drawer
# python train_PEBBLE.py \
#     env=metaworld_drawer-open-v2 \
#     seed=0 \
#     exp_name=reproduce \
#     reward=learn_from_preference \
#     vlm_label=1 \
#     vlm=gemini_free_form \
#     image_reward=1 \
#     reward_batch=40 \
#     segment=1 \
#     teacher_eps_mistake=0 \
#     reward_update=10 \
#     num_interact=4000 \
#     max_feedback=20000 \
#     agent.params.actor_lr=0.0003 agent.params.critic_lr=0.0003 gradient_update=1 activation=tanh num_unsup_steps=9000 \
#     num_train_steps=1000000 agent.params.batch_size=512 double_q_critic.params.hidden_dim=256 double_q_critic.params.hidden_depth=3 \
#     diag_gaussian_actor.params.hidden_dim=256 diag_gaussian_actor.params.hidden_depth=3  \
#     feed_type=0 teacher_beta=-1 teacher_gamma=1  teacher_eps_skip=0 teacher_eps_equal=0 \
#     num_eval_episodes=1 \
#     cached_label_path=data/cached_labels/Drawer/seed_0/


# ### sweep into
# python train_PEBBLE.py \
#     env=metaworld_sweep-into-v2 \
#     seed=0 \
#     exp_name=reproduce \
#     reward=learn_from_preference \
#     vlm_label=1 \
#     vlm=gemini_free_form \
#     image_reward=1 \
#     reward_batch=40 \
#     segment=1 \
#     teacher_eps_mistake=0 \
#     reward_update=10 \
#     num_interact=4000 \
#     max_feedback=20000 \
#     agent.params.actor_lr=0.0003 agent.params.critic_lr=0.0003 gradient_update=1 activation=tanh num_unsup_steps=9000 \
#     num_train_steps=1000000 agent.params.batch_size=512 double_q_critic.params.hidden_dim=256 double_q_critic.params.hidden_depth=3 \
#     diag_gaussian_actor.params.hidden_dim=256 diag_gaussian_actor.params.hidden_depth=3  \
#     feed_type=0 teacher_beta=-1 teacher_gamma=1  teacher_eps_skip=0 teacher_eps_equal=0 \
#     num_eval_episodes=1 \
#     cached_label_path=data/cached_labels/Sweep-Into/combined/

# ### cartpole
# python train_PEBBLE.py \
#     env=CartPole-v1 \
#     seed=0 \
#     exp_name=reproduce \
#     reward=learn_from_preference \
#     vlm=gemini_free_form \
#     vlm_label=1 \
#     exp_name=2024-3-24-icml-rebuttal-more-seeds \
#     segment=1 \
#     image_reward=1 \
#     max_feedback=10000 reward_batch=50 reward_update=50 \
#     num_interact=5000 \
#     num_train_steps=500000 \
#     agent.params.critic_lr=0.0005 gradient_update=1 activation=tanh num_unsup_steps=1000 num_train_steps=500000   \
#     feed_type=0 teacher_beta=-1 teacher_gamma=1 teacher_eps_mistake=0 teacher_eps_skip=0 teacher_eps_equal=0  \
#     agent.params.actor_lr=0.0005 \
#     cached_label_path=data/cached_labels/CartPole/seed_0/

# # RopeFlattenEasy
# python train_PEBBLE.py \
#     env=softgym_RopeFlattenEasy \
#     seed=0 \
#     exp_name=reproduce \
#     reward=learn_from_preference \
#     vlm_label=1 \
#     vlm=gemini_free_form \
#     image_reward=1 \
#     reward_batch=100 \
#     segment=1 \
#     teacher_eps_mistake=0 \
#     reward_update=30 \
#     num_interact=5000 \
#     max_feedback=20000 \
#     reward_lr=1e-4 \
#     agent.params.actor_lr=0.0003 agent.params.critic_lr=0.0003 gradient_update=1 activation=tanh num_unsup_steps=9000 \
#     num_train_steps=600000 agent.params.batch_size=512 double_q_critic.params.hidden_dim=256 double_q_critic.params.hidden_depth=3 \
#     diag_gaussian_actor.params.hidden_dim=256 diag_gaussian_actor.params.hidden_depth=3  \
#     feed_type=0 teacher_beta=-1 teacher_gamma=1  teacher_eps_skip=0 teacher_eps_equal=0 \
#     num_eval_episodes=1 \
#     resnet=1 \
#     cached_label_path=data/cached_labels/RopeFlattenEasy/seed_0/

# # PassWater
# python train_PEBBLE.py \
#     env=softgym_PassWater \
#     seed=0 \
#     exp_name=reproduce \
#     reward=learn_from_preference \
#     vlm_label=1 \
#     vlm=gemini_free_form \
#     image_reward=1 \
#     reward_batch=100 \
#     segment=1 \
#     teacher_eps_mistake=0 \
#     reward_update=30 \
#     num_interact=5000 \
#     max_feedback=20000 \
#     reward_lr=1e-4 \
#     agent.params.actor_lr=0.0003 agent.params.critic_lr=0.0003 gradient_update=1 activation=tanh num_unsup_steps=9000 \
#     num_train_steps=600000 agent.params.batch_size=512 double_q_critic.params.hidden_dim=256 double_q_critic.params.hidden_depth=3 \
#     diag_gaussian_actor.params.hidden_dim=256 diag_gaussian_actor.params.hidden_depth=3  \
#     feed_type=0 teacher_beta=-1 teacher_gamma=1  teacher_eps_skip=0 teacher_eps_equal=0 \
#     num_eval_episodes=1 \
#     resnet=1 \
#     cached_label_path=data/cached_labels/PassWater/seed_0/

