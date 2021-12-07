### training TTSR
train_name='TTSR_AfterHyperparameterTuning'
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
               --dataset_dir ../../datasets/32-128_1024-4096_fullSet_noSSIMTreshold\
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
               --tpl_use_S False\
               --batch_size 10\
               --num_init_epochs 0\
               --num_epochs 15\
               --print_every 5\
               --save_every 3\
               --val_every 3\
               --gray True\
               --gray_transform True\
               --GAN_type WGAN_GP\
               --train_crop_size 32\
               --debug True\
               --ref_crop_size 0\
               --ref_image_size 300\
               --NumbRef 500\
               --retrain True\
               --model_path ./pretrainedModels/model_00015.pt
               
               #batchsize 10
               
#BATCH SIZE: 10 FOR REF_CROP_SIZE = 10         --save_dir ./train/IMM/TTSR/$save_folder/       
#               --dataset_dir /home/ps815691/datasets/32-128_1024-4096_fullSet_noSSIMTreshold\
# Small32-128_1024-4096_SSIMTreshold0-0
# ### training TTSR-rec
# python main.py --save_dir ./train/CUFED/TTSR-rec \
#                --reset True \
#                --log_file_name train.log \
#                --num_gpu 1 \
#                --num_workers 9 \
#                --dataset CUFED \
#                --dataset_dir /home/v-fuyang/Data/CUFED/ \
#                --n_feats 64 \
#                --lr_rate 1e-4 \
#                --lr_rate_dis 1e-4 \
#                --lr_rate_lte 1e-5 \
#                --rec_w 1 \
#                --per_w 0 \
#                --tpl_w 0 \
#                --adv_w 0 \
#                --batch_size 9 \
#                --num_init_epochs 0 \
#                --num_epochs 200 \
#                --print_every 600 \
#                --save_every 10 \
#                --val_every 10