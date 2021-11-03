### evaluation
train_name='EvaluationHeatmapTests'
curdate=$(date +'%Y_%m_%d-%H_%M')
save_folder="${curdate}_${train_name}"
python main.py --save_dir ./eval/IMM/TTSR/$save_folder/ \
               --reset True \
               --log_file_name eval.log \
               --eval True \
               --eval_save_results True \
               --num_workers 4 \
               --dataset IMMRW \
               --dataset_dir /home/ps815691/datasets/32-128_1024-4096_fullSet_noSSIMTreshold/ \
               --model_path ./train/IMM/TTSR/Init_Training/2021_11_01-12_42_RW_TTSR_RSM_training_init_CyclicLR_LTElr_e-4/model/best_model_loss_4.pt\
               --gray True \
               --gray_transform True \
               --cpu False\
               --debug True\
               --rec_w 1e-3 \
               --per_w 1e-2 \
               --tpl_w 1e-2 \
               --adv_w 1e-3 \
               --train_crop_size 32\
               --ref_crop_size 0\
               --ref_image_size 300
               