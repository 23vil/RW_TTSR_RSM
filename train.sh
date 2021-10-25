### training TTSR
train_name='Test_fullDataset_Epochtime'
curdate=$(date +'%Y_%m_%d-%H_%M')
save_folder="${curdate}_${train_name}"
python main.py --save_dir ./train/IMM/TTSR/$save_folder/\
               --training_name $train_name\
               --reset True\
               --log_file_name train.log\
               --num_gpu 1\
               --cpu True\
               --num_workers 9\
               --dataset IMMRW\
               --dataset_dir /home/ps815691/datasets/Small32-128_1024-4096_SSIMTreshold0-0/\
               --n_feats 64\
               --lr_rate 1e-4\
               --lr_rate_dis 1e-4\
               --lr_rate_lte 1e-4\
               --rec_w 1e-3\
               --per_w 1e-2\
               --tpl_w 1e-2\
               --adv_w 1e-3\
               --GAN_k 9\
               --tpl_use_S True\
               --batch_size 1\
               --num_init_epochs 0 \
               --num_epochs 25\
               --print_every 1\
               --save_every 1\
               --val_every 1\
               --gray True\
               --gray_transform True\
               --GAN_type GAN\
               --train_crop_size 32\
               --debug True\
               --refTrain True\
               --seperateRefLoss False\
               --ref_crop_size 300\
               --ref_image_size 0\
               --retrain False\
               --model_path ./pretrainedModels/model_00001.pt
               #batchsize 10
#BATCH SIZE: 10 FOR REF_CROP_SIZE = 10             
#               --dataset_dir /home/ps815691/datasets/32-128_1024-4096_SSIMTreshold0-1 \
# /home/ps815691/git/TTSR/dataset/IMMRW_64_256
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