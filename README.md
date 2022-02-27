<!-- OMP_NUM_THREADS=1 MKL_NUM_THREADS=1  -->
CUDA_VISIBLE_DEVICES=7,6 python main.py --n_worker 16 --save_replay_buffer --td3.expl_noise 0.2 --td3.policy_noise 0.3 --td3.clip_noise 0.5 --td3.nstep 3 --td3.knn_k 512 --td3.knn_avg True --batch_size 1024 --experiment_id k512_3step_100M --replay_buffer_size 100000001 --n_epochs 100000000 --replay_dir /shared/hao/dataset/big  --online

CUDA_VISIBLE_DEVICES=7,6 python main.py --n_worker 16 --save_replay_buffer --td3.expl_noise 0.2 --td3.policy_noise 0.3 --td3.clip_noise 0.5 --td3.nstep 3 --td3.knn_k 3 --td3.knn_avg True --batch_size 1024 --experiment_id k3_3step_100M --replay_buffer_size 100000001 --n_epochs 100000000 --replay_dir /shared/hao/dataset/big --online


CUDA_VISIBLE_DEVICES=7,6 python main.py --n_worker 16 --save_replay_buffer --td3.expl_noise 0.2 --td3.policy_noise 0.3 --td3.clip_noise 0.5 --td3.nstep 3 --td3.knn_k 512 --td3.knn_avg True --batch_size 1024 --experiment_id k512_3step_100M --replay_buffer_size 100000001 --n_epochs 100000000 --replay_dir /shared/hao/dataset/big --pin_memory True --online
