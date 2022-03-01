<!-- OMP_NUM_THREADS=1 MKL_NUM_THREADS=1  -->

CUDA_VISIBLE_DEVICES=3,2 python main.py --n_worker 16 --td3.expl_noise 0.2 --td3.policy_noise 0.3 --td3.clip_noise 0.5 --td3.nstep 3 --td3.knn_k 512 --td3.knn_clip 0.1 --td3.knn_avg True --batch_size 1024 --experiment_id k512_3step_100M_c --n_epochs 100000000 --eval_period 100000 --replay_dir /home/hao/dataset/big --online

CUDA_VISIBLE_DEVICES=1,0 python main.py --n_worker 16 --td3.expl_noise 0.2 --td3.policy_noise 0.3 --td3.clip_noise 0.5 --td3.nstep 3 --td3.knn_k 512 --td3.knn_clip 0.1 --td3.knn_avg True --batch_size 1024 --experiment_id k512_3step_100M_c --n_epochs 100000 --eval_period 100 --n_train_step_per_epoch 1000 --n_sample_step_per_epoch 1000 --replay_dir /home/hao/dataset/big --online
