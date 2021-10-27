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
               --dataset_dir /home/ps815691/datasets/Small32-128_1024-4096_SSIMTreshold0-0/ \
               --model_path ./pretrainedModels/model_00001.pt\
               --gray True \
               --gray_transform True \
               --cpu True\
               --debug True\
               --rec_w 1 \
               --per_w 1e-2 \
               --tpl_w 1e-2 \
               --adv_w 1e-3 \
               --train_crop_size 32
               