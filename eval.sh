### evaluation
train_name='EvaluationMgALCa'
curdate=$(date +'%Y_%m_%d-%H_%M')
save_folder="${train_name}_${curdate}"
python main.py --save_dir ./eval/IMM/TTSR/$save_folder/ \
               --reset True \
               --log_file_name eval.log \
               --eval True \
               --eval_save_results True \
               --num_workers 4 \
               --dataset IMMRW \
               --dataset_dir /home/ps815691/datasets/infoDP800_RW/DP800_768x768_to_3072x3072_32 \
               --reference_dir ../../datasets/backup/ReferenceImages\
               --model_path /work/ps815691/trainingResults/2022_01_14-05_47_TTSR_Tutorial/model/init-model_00001.pt\
               --discr_path /work/ps815691/trainingResults/ModelA_HT+15/model/discriminator_00015.pt\
               --gray True \
               --gray_transform True \
               --cpu False\
               --debug False\
               --rec_w 1e-3 \
               --per_w 1e-2 \
               --tpl_w 1e-4 \
               --adv_w 1e-4 \
               --train_crop_size 32\
               --ref_image_size 300
               