step 1. 训练VQ-VAE
train_vq_clip8.py --total-iter 20000 --lr-scheduler 5000 12000 --batch-size 1024 --nb-code 1024 --code-dim 1024 --width 1024 --output-emb-width 1024 --window_size 64 --windows 3 --exp-name mo266/smplxmo266_0401_win64_Norm --dataset finedance --gpu 7 --loss-vel 0.1 --lr 1e-4 --feature_dim 266 --print-iter=10 --save-iter 10 --recons-loss l1 --partial full


step 2. Test VQ-VAE

python test_vq_clip8_flat.py --exp-name mo266_train/FineDance_0328_1024_win128 --nb-code 1024 --code-dim 1024 --width 1024 --output-emb-width 1024 --dataset finedance --gpu 2 --resume-pth experiments/vqvae/output_vq/mo266/smplxmo266_0401_win64_Norm/vqvae5870/f_266/best_recon.pth  --feature_dim 266 --exp-name mo266/smplxmo266_0401_win64_Norm



step 3. Train GPT
VQ-VAE window 为64 训练 GPT windows_size 128

python train_gpt_mask.py --batch-size 128 --total-iter 100000 --lr-scheduler 15000 400000  --window_size 128 --windows 20 --nb-code 1024  --input-dim 2 --gpu 1 --exp-name mo266/smplxmo266_0401_win64_Norm   --tk_dir  /data/lrh/project/dance/LongV2/LongV2Exp1/experiments/vqvae/output_vq/mo266/smplxmo266_0401_win64_Norm/eval_idx_flat

