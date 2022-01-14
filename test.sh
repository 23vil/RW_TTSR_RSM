### test
test_name='DP_1024_32x32_MOdelB_WiTH_mOre_imAGes'
curdate=$(date +'%Y_%m_%d-%H_%M')
save_folder="${curdate}_${test_name}"
#./test/demo/output/$save_folder/
python main.py --save_dir ./test/demo/output/$save_folder/ \
			   --reset True \
			   --log_file_name test.log \
			   --test True \
			   --num_workers 3 \
			   --lr_path ./test/demo/lr/2.tif \
                           --ref_image_size 300\
                           --NumbRef 500\
			   --gray True\
			   --gray_transform True\
			   --cpu False\
			   --model_path /work/ps815691/trainingResults/2022_01_14-05_47_TTSR_Tutorial/model/init-model_00001.pt\
			   --reference_dir ../../datasets/backup/ReferenceImages
			   #--model_path /work/ps815691/trainingResults/MOdelB_HT_+6+15/model/model_00009.pt
			   
