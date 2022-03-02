<!-- OMP_NUM_THREADS=1 MKL_NUM_THREADS=1  -->

python main.py --n_worker 16 --td3.expl_noise 0.2 --td3.policy_noise 0.3 --td3.clip_noise 0.5 --td3.nstep 3 --td3.knn_k 512 --td3.knn_clip 0.1 --td3.knn_avg True --td3.knn_pow 2 --batch_size 1024 --experiment_id 1vs1 --n_epochs 100000000 --eval_period 100000 --n_train_step_per_epoch 1 --n_sample_step_per_epoch 1 --replay_dir /home/hao/dataset/big --online


python main.py --n_worker 16 --td3.expl_noise 0.2 --td3.policy_noise 0.3 --td3.clip_noise 0.5 --td3.nstep 3 --td3.knn_k 512 --td3.knn_clip 0.1 --td3.knn_avg True --td3.knn_pow -1 --batch_size 1024 --experiment_id 1vs1-1 --n_epochs 100000000 --eval_period 100000 --n_train_step_per_epoch 1 --n_sample_step_per_epoch 1 --replay_dir /home/hao/dataset/big --online


python main.py --n_worker 16 --td3.expl_noise 0.2 --td3.policy_noise 0.3 --td3.clip_noise 0.5 --td3.nstep 3 --td3.knn_k 512 --td3.knn_clip 0.1 --td3.knn_avg True --td3.knn_pow 2 --batch_size 1024 --experiment_id 1kvs1k --n_epochs 100000 --eval_period 100 --n_train_step_per_epoch 1000 --n_sample_step_per_epoch 1000 --replay_dir /home/hao/dataset/big --online


python main.py --n_worker 16 --td3.expl_noise 0.2 --td3.policy_noise 0.3 --td3.clip_noise 0.5 --td3.nstep 3 --td3.knn_k 512 --td3.knn_clip 0.1 --td3.knn_avg True --td3.knn_pow -1 --batch_size 1024 --experiment_id 1kvs1k-1 --n_epochs 100000 --eval_period 100 --n_train_step_per_epoch 1000 --n_sample_step_per_epoch 1000 --replay_dir /home/hao/dataset/big --online
