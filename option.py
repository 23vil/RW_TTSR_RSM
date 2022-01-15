import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='TTSR')

parser.add_argument('--training_name', type=str, default='TTSR',
                    help='Short description of training goal.')

### log setting
parser.add_argument('--save_dir', type=str, default='save_dir',
                    help='Directory to save log, arguments, models and images')
parser.add_argument('--reset', type=str2bool, default=False,
                    help='Delete save_dir to create a new one')
parser.add_argument('--log_file_name', type=str, default='TTSR.log',
                    help='Log file name')
parser.add_argument('--logger_name', type=str, default='TTSR',
                    help='Logger name')

### device setting
parser.add_argument('--cpu', type=str2bool, default=False,
                    help='Use CPU to run code')
parser.add_argument('--num_gpu', type=int, default=1,
                    help='The number of GPU used in training')
parser.add_argument('--cluster', type=str2bool, default='False',
                    help='Depending on the operating system, that this code is run on, file links are interpreted differently. By setting --cluster to true, the state that I could run on RWTHs cluster is used. --cluster = False can be run on a local machine')

### dataset setting
parser.add_argument('--dataset', type=str, default='IMMRW',
                    help='Which dataset to train and test')
parser.add_argument('--dataset_dir', type=str, default='/home/ps815691/datasets/Small32-128_1024-4096_SSIMTreshold0-0/',
                    help='Directory of dataset')
parser.add_argument('--gray', type=str2bool, default='True',
                    help='True if dataset consists of grayscale images')
parser.add_argument('--gray_transform', type=str2bool, default='True',
                    help='If True, transforms SR output from RGB to 3-Channel Grayscale, before calculating loss')

### reference selection model settings
parser.add_argument('--reference_dir', type=str, default='/home/ps815691/datasets/ReferenceImages',
                    help='Directory of reference images')
parser.add_argument('--NumbRef', type=int, default=500,
                    help='Number of available ReferenceImages')
parser.add_argument('--ref_crop_size', type=int, default='0',
                    help='Uses random crops of ref_crop_size*ref_crop_size of the original reference image for training etc. If 0 --> Original image size (ref_image_size) is used.')
parser.add_argument('--ref_image_size', type=int, default='0',
                    help='Has to be set if ref_crop_size is 0.')



### dataloader setting
parser.add_argument('--num_workers', type=int, default=8,
                    help='The number of workers when loading data')

### model setting
parser.add_argument('--num_res_blocks', type=str, default='16+16+8+4',
                    help='The number of residual blocks in each stage')
parser.add_argument('--n_feats', type=int, default=64,
                    help='The number of channels in network')
parser.add_argument('--res_scale', type=float, default=1.,
                    help='Residual scale')

### loss setting
parser.add_argument('--GAN_type', type=str, default='WGAN_GP',
                    help='The type of GAN used in training')
parser.add_argument('--GAN_k', type=int, default=2,
                    help='Training discriminator k times when training generator once')
parser.add_argument('--tpl_use_S', type=str2bool, default=False,
                    help='Whether to multiply soft-attention map in transferal perceptual loss')
parser.add_argument('--tpl_type', type=str, default='l2',
                    help='Which loss type to calculate gram matrix difference in transferal perceptual loss [l1 / l2]')
parser.add_argument('--rec_w', type=float, default=1.,
                    help='The weight of reconstruction loss') # cannot be smaller than 1e-8, as defined in loss.py
parser.add_argument('--per_w', type=float, default=1e-2,
                    help='The weight of perceptual loss')  # cannot be smaller than 1e-8, as defined in loss.py
parser.add_argument('--tpl_w', type=float, default=1e-2,
                    help='The weight of transferal perceptual loss')  # cannot be smaller than 1e-8, as defined in loss.py
parser.add_argument('--adv_w', type=float, default=1e-3,
                    help='The weight of adversarial loss')  # cannot be smaller than 1e-8, as defined in loss.py

### optimizer setting
parser.add_argument('--beta1', type=float, default=0.9,
                    help='The beta1 in Adam optimizer')
parser.add_argument('--beta2', type=float, default=0.999,
                    help='The beta2 in Adam optimizer')
parser.add_argument('--eps', type=float, default=1e-8,
                    help='The eps in Adam optimizer')
parser.add_argument('--lr_rate', type=float, default=1e-4,
                    help='Learning rate in cas Step_LR is activated')
parser.add_argument('--lr_base', type=float, default=1e-4,
                    help='Lower boundary Learning rate cyclicLR')
parser.add_argument('--lr_max', type=float, default=1e-4,
                    help='Lower boundary Learning rate cyclicLR')
parser.add_argument('--lr_rate_dis_fkt', type=float, default=2,
                    help='Learning rate of discriminator')
parser.add_argument('--lr_rate_lte', type=float, default=1e-5,
                    help='Learning rate of LTE')
parser.add_argument('--decay', type=float, default=1,
                    help='Learning rate decay type(STepLR)')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='Learning rate decay factor for step decay(STepLR)')

### training setting
parser.add_argument('--batch_size', type=int, default=1,
                    help='Training batch size')
parser.add_argument('--train_crop_size', type=int, default=40,
                    help='Training data crop size')   #TRAIN CROP Size!!
parser.add_argument('--num_init_epochs', type=int, default=2,
                    help='The number of init epochs which are trained with only reconstruction loss')
parser.add_argument('--num_epochs', type=int, default=1,
                    help='The number of training epochs')
parser.add_argument('--print_every', type=int, default=1,
                    help='Print period')
parser.add_argument('--save_every', type=int, default=999999,
                    help='Save period')
parser.add_argument('--val_every', type=int, default=999999,
                    help='Validation period')
parser.add_argument('--retrain', type=str2bool, default=False,
                    help='Whether load old model and train it again (True), or train new model (False).')
parser.add_argument('--SaveOptimAndDiscr', type=str2bool, default=False,
                    help='Optimizer and Discriminator will be saved.')
parser.add_argument('--LoadOptimAndDiscr', type=str2bool, default=False,
                    help='Optimizer and Discriminator will be loaded from files.')
parser.add_argument('--optim_path', type=str, default=None,
                    help='Path to saved TTSR model optimizer')
parser.add_argument('--discr_path', type=str, default=None,
                    help='Path to saved TTSR Discriminator')
parser.add_argument('--discr_optim_path', type=str, default=None,
                    help='Path to saved TTSR Discriminators Optimizer')



### evaluate / test / finetune setting
parser.add_argument('--eval', type=str2bool, default=False,
                    help='Evaluation mode')
parser.add_argument('--eval_save_results', type=str2bool, default=False,
                    help='Save each image during evaluation')
parser.add_argument('--model_path', type=str, default=None,
                    help='The path of model to evaluation')
parser.add_argument('--test', type=str2bool, default=False,
                    help='Test mode')
parser.add_argument('--lr_path', type=str, default=None,
                    help='The path of input lr image when testing')
parser.add_argument('--lr_folder', type=str, default=None,
                    help='From all images within this folder test SR images will be generated')

parser.add_argument('--ref_path', type=str, default='./test/demo/ref/ref.png',
                    help='The path of ref image when testing')
parser.add_argument('--debug', type=str2bool, default=False,
                    help='Prints a loss heatmap, that enables an analysation for which image parts lead to a high loss value.')

args = parser.parse_args()
