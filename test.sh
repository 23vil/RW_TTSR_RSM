### test
save_folder=$(date +'%Y_%m_%d-%H_%M')
python main.py --save_dir ./test/demo/output/$save_folder/ \
			   --reset True \
			   --log_file_name test.log \
			   --test True \
			   --num_workers 3 \
			   --lr_path ./test/demo/lr/82.tif \
			   --ref_crop_size 0\
                           --ref_image_size 300\
                           --NumbRef 500\
			   --model_path ./pretrainedModels/best_model_loss_4.pt \
			   --gray True\
			   --gray_transform True\
			   --cpu False
