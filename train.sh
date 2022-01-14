### training TTSR
train_name='TTSR_Tutorial'
curdate=$(date +'%Y_%m_%d-%H_%M')
save_folder="${curdate}_${train_name}"
python main.py --save_dir ../../../../work/ps815691/trainingResults/$save_folder\
               --training_name $train_name\
               --reset True\
               --log_file_name train.log\
               --num_gpu 1\
               --cpu False\
               --num_workers 9\
               --dataset IMMRW\
               --dataset_dir ../../datasets/infoDP800_RW/DP800_768x768_to_3072x3072_32\
               --reference_dir ../../datasets/backup/ReferenceImages\
               --n_feats 64\
               --lr_rate 1e-5\
               --lr_max 6.4e-5\
               --lr_base 1e-8\
               --lr_rate_dis_fkt 1\
               --lr_rate_lte 1e-4\
               --rec_w 1e-3\
               --per_w 1e-2\
               --tpl_w 1e-4\
               --adv_w 1e-4\
               --GAN_k 10\
               --batch_size 2\
               --num_init_epochs 1\
               --num_epochs 0\
               --print_every 1\
               --save_every 1\
               --val_every 3\
               --gray True\
               --GAN_type WGAN_GP\
               --train_crop_size 32\
               --debug True\
               --ref_crop_size 0\
               --ref_image_size 300\
               --NumbRef 500\
               --retrain False\
               --model_path /work/ps815691/trainingResults/ModelA_HT+15/model/model_00015.pt\
               --SaveOptimAndDiscr False\
               --LoadOptimAndDiscr False\
               --optim_path /work/ps815691/trainingResults/ModelA_HT+15/model/optimizer_00001.pt\
               --discr_path /work/ps815691/trainingResults/ModelA_HT+15/model/discriminator_00015.pt\
               --discr_optim_path /work/ps815691/trainingResults/ModelA_HT+15/model/discriminator_optim_00015.pt
               
               #batchsize 10
               
#BATCH SIZE: 10 FOR REF_CROP_SIZE = 10         --save_dir ./train/IMM/TTSR/$save_folder/       
#               --dataset_dir /home/ps815691/datasets/32-128_1024-4096_fullSet_noSSIMTreshold\
# Small32-128_1024-4096_SSIMTreshold0-0
