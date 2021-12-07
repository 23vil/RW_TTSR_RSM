### test
save_folder=$(date +'%Y_%m_%d-%H_%M')
#./test/demo/output/$save_folder/
python main.py --save_dir ./test/demo/output/$save_folder/ \
			   --reset True \
			   --log_file_name test.log \
			   --test True \
			   --num_workers 3 \
			   --lr_path ./test/demo/lr/lr44b.tif \
			   --ref_crop_size 0\
                           --ref_image_size 300\
                           --NumbRef 500\
			   --model_path ./pretrainedModels/model_00015.pt \
			   --gray True\
			   --gray_transform True\
			   --cpu False

			   #--model_path ./pretrainedModels/best_model_psnr_12.pt \
			   #./pretrainedModels/model_00015.pt