### test
save_folder=$(date +'%Y_%m_%d-%H_%M')
python main.py --save_dir ./test/demo/output/$save_folder/ \
			   --reset True \
			   --log_file_name test.log \
			   --test True \
			   --num_workers 3 \
			   --lr_path ./test/demo/lr/005008_42.tif \
			   --ref_path ./test/demo/ref/a2.tif \
			   --model_path ./pretrainedModels/complete_model_00001.pt \
			   --ref_model_path ./pretrainedModels/ref_model_00000.pt\
			   --gray True\
			   --gray_transform True\
			   --cpu True\
			   --seperateRefLoss False
