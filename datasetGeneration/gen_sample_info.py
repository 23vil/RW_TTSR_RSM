"""
gen_sample_info.py
-------
TTSR - Dataset generation and extraction of metadata   ---- STEP 1

This script marks the first step in the generation of a dataset for the SEM_TTSR model. It walks through files in your
datasources and extracts the metadata of each image, and will serve as a guide map for training data generation. Datasources 
can have multiple subcategories and have to be devided into hr and lr images. Both hr and lr images themselves need to be further
split into training and validation data.


The datasource folder has to follow the hierarchy as below:
- datasources
  |- fixed
     |- subcategory1
     |  |- hr
     |  |   |- train
     |  |   |   |- img1.tif
     |  |   |   |- img2.tif
     |  |   |   |- ...
     |  |   |- valid
     |  |   |   |- img3.tif
     |  |   |   |- img4.tif
     |  |   |   |- ...
     |  |- lr
     |  |   |- train
     |  |   |   |- img1.tif
     |  |   |   |- img2.tif
     |  |   |   |- ...
     |  |   |- valid
     |  |   |   |- img3.tif
     |  |   |   |- img4gggggggggggggggggggggggg.tif
     |  |   |   |- ...
     |
     |- subcategory2
     |  |- hr
     |  |   |- train
     |  |   |   |- img1.tif
     |  |   |   |- img2.tif
     |  |   |   |- ...
     |  |   |- valid
     |  |   |   |- img1.tif
     |  |   |   |- img2.tif
     |  |   |   |- ...
     |  |- lr
     |  |   |- train
     |  |   |   |- img1.tif
     |  |   |   |- img2.tif
     |  |   |   |- ...
     |  |   |- valid
     |  |   |   |- img1.tif
     |  |   |   |- img2.tif
     |  |   |   |- ...
     |- ... 
Notes:
-------
1. Except folders named 'fixed', 'hr', 'lr', 'train' or 'valid',
all other folders and files can be changed to different names accrodingly.
2. .tif/.tiff images are supported in RGB and grayscale. Images with alpha channel are not supported.

Parameters:
-------
- out: path, output csv file name
- sources: path, whitelist all datasource folders to include, if more than one
           need to be included. (optional)
- only: str, whitelist subfolders to include. This is useful if only a few
        subfolders among a large number of subfolders need to be included. (optional)
- skip: str, subfolders to skip. This is useful when most of the subfolders
        need to be included except a few. (optional)

Returns:
-------
- csv file: A csv file that saves useful metadata of images in the datasource.
  Each frame in each image is saved as one row separately, and each row has 24
  columns of information detailed as follows:
        - 'category': str, data category. This is useful when we want to generate a
                        'combo' dataset mixed with different biological structures, for
                        example, mitochondria and microtubules.
        - 'dsplit': str, 'train' if the source image is in the folder 'train', and
                    'valid' if the source image is in the folder 'valid'.

        - 'uint8': boolean, 'TRUE' or 'FALSE'. It is 'TRUE' if the source file of this
                    frame is 8-bit unsigned interger, otherwise 'FALSE'.
        - 'mean': float, mean value of the frame.
        - 'sd': float, standard deviation of the frame.
        - 'all_rmax': int, maximum value of the whole source image stack.
        - 'all_mi': int, 2 percentile value of the whole source image stack.
        - 'all_ma': int, 99.99 percentile value of the whole source image stack.
        - 'mi': int, 2 percentile value of the frame.
        - 'ma': int, 99.99 percentile value of the frame.
        - 'rmax': int, 255 if the source stack is 8-bit unsigned interger, otherwise
                    it is the maximum value of the whole source image stack.
        - 'nc': int, number of channels of the source image stack.
        - 'c': int, channel number of this frame. The first channel starts counting at 0.
        - 'x': int, dimension in X.
        - 'y': int, dimension in Y.
        - 'fn': str, relative path of the source image.

Examples:
-------
Following are a couple of examples showing how to generate the metadata
from the datasource in different ways. Given the datasource folder is strctured
as below:
- datasources
  |- fixed
     |- DP800_1024x1024_to_4096x4096
     |  |- hr
     |  |   |- train
     |  |   |   |- hr1.tif
     |  |   |   |- hr2.tif
     |  |   |- valid
     |  |   |   |- hr3.tif
     |  |   |   |- hr4.tif
     |  |- lr
     |  |   |- train
     |  |   |   |- lr1.tif
     |  |   |   |- lr2.tif
     |  |   |- valid
     |  |   |   |- lr3.tif
     |  |   |   |- lr4.tif
     |
     |- MgAlCa_1024x1024_to_4096x4096
     |  |- hr
     |  |   |- train
     |  |   |   |- hr1.tif
     |  |   |   |- hr2.tif
     |  |   |- valid
     |  |   |   |- hr3.tif
     |  |   |   |- hr4.tif
     |  |- lr
     |  |   |- train
     |  |   |   |- lr1.tif
     |  |   |   |- lr2.tif
     |  |   |- valid
     |  |   |   |- lr3.tif
     |  |   |   |- lr4.tif



-!-!-!-!-!-!-!-!-!-!-!-! How to start the script: -!-!-!-!-!-!-!-!-!-!-!-!

Example 1:
Only 'DP800_1024x1024_to_4096x4096' folder in datasource is needed. Name output
.csv file as 'DP800.csv'
-->python gen_sample_info.py --only DP800_1024x1024_to_4096x4096 --out DP800.csv /home/datasources/fixed

Example 2:
All subcategories in datasource are needed for training. Name
output file as 'all.csv'.
-->python gen_sample_info.py --out all.csv /home/datasources/fixed

Example 3:
All subcategroies in datasource  are needed for training except
files in 'DP800_1024x1024_to_4096x4096'. Name output file as 'AllButDP.csv'
-->python gen_sample_info.py --skip DP800_1024x1024_to_4096x4096 --out AllButDP.csv.csv /home/datasources/fixed

"""
import yaml
from fastai.script import *
import pandas as pd
import numpy as np
from utils import *
from pathlib import Path
from time import sleep
import shutil
import PIL
import czifile

PIL.Image.MAX_IMAGE_PIXELS = 99999999999999

def process_tif(item, category, mode):
    #item is tuple (hr,lr)
    hrItem = item[0]
    lrItem = item[1]  
    tif_srcs = []
    
    #hr
    hrImg = PIL.Image.open(hrItem)
    h_x,h_y = hrImg.size

    
    #lr
    lrImg = PIL.Image.open(lrItem)
    l_x,l_y = lrImg.size



    #This is only done for HR data------------
    HRdata = []
    HRimg_data = np.array(hrImg)
    HRdata.append(HRimg_data)
    HRdata = np.stack(HRdata)
    HRall_rmax = HRdata.max()
    HRall_mi, HRall_ma = np.percentile(HRdata, [2,99.99])
    
    
    
    #This is only done for LR data------------
    LRdata = []
    LRimg_data = np.array(lrImg)
    LRdata.append(LRimg_data)
    LRdata = np.stack(LRdata)
    LRall_rmax = LRdata.max()
    LRall_mi, LRall_ma = np.percentile(LRdata, [2,99.99])


    #HR
    HRimg_data = HRdata#[n]
    HRdtype = HRimg_data.dtype
    HRmi, HRma = np.percentile(HRimg_data, [2,99.99])
    if HRdtype == np.uint8: HRrmax = 255.
    else: HRrmax = HRimg_data.max()
            
            
    #LR
    LRimg_data = LRdata#[n]
    LRdtype = LRimg_data.dtype
    LRmi, LRma = np.percentile(LRimg_data, [2,99.99])
    if LRdtype == np.uint8: LRrmax = 255.
    else: LRrmax = LRimg_data.max()

    tif_srcs.append({'HRfn': hrItem, 'LRfn': lrItem, 'ftype': 'tif', 'category': category, 'dsplit': mode,
                         'uint8': HRdtype==np.uint8, 'HRmi': HRmi, 'HRma': HRma, 'HRrmax': HRrmax,
                         'HRall_rmax': HRall_rmax, 'HRall_mi': HRall_mi, 'HRall_ma': HRall_ma,
                         'HRmean': HRimg_data.mean(), 'HRsd': HRimg_data.std(),
                         'nc': 1,
                         'c':0, 'HRx': h_x, 'HRy': h_y,
                         'LRmi': LRmi, 'LRma': LRma, 'LRrmax': LRrmax,
                         'LRall_rmax': LRall_rmax, 'LRall_mi': LRall_mi, 'LRall_ma': LRall_ma,
                         'LRmean': LRimg_data.mean(), 'LRsd': LRimg_data.std(),
                         'LRx': l_x, 'LRy': l_y})
    return tif_srcs

def process_unk(item, category, mode):
    print(f"**** Unknown: {item}")
    return []

def process_item(item, category, mode): #for .tif returns function process_tif(item, category, mode)
    try:
        if mode == 'test': return []
        else:
            item_map = {
                '.tif': process_tif,
                '.tiff': process_tif
            }
            
            map_f = item_map.get(item[0].suffix, process_unk)# item.suffix is ending of items filepath (e.g. .tif) ... item is a tuple containing hr & lr
            return map_f(item, category, mode) #map_f is either process_tif 
    except Exception as ex:
        print(f'err procesing: {item}')
        print(ex)
        return []

def build_tifs(src):#, mbar=None):
    tif_srcs = []
    for mode in ['train', 'valid', 'test']:
        hr_src_dir = src / "hr" / mode
        lr_src_dir = src / "lr" / mode
        category = src.stem #returns final path component without suffix, so without .tif ---> means filename or folder name --> Dataset Name
        hr_items = list(hr_src_dir.iterdir()) if hr_src_dir.exists() else []
        lr_items = list(lr_src_dir.iterdir()) if lr_src_dir.exists() else []
        items = list(zip(hr_items, lr_items)) if len(hr_items) == len(lr_items) and lr_items != [] and hr_items != [] else []

        for p in items:
            tif_srcs += process_item(p, category=category, mode=mode)       
                
                
    return tif_srcs

@call_parse
def main(out: Param("tif output name", Path, required=True),
         sources: Param('src folders', Path, nargs='...', opt=False) = None, #folder: datasources/fixed   data is in datasources/fixed/IMMresVar2048-4096/hr/train
         only: Param('whitelist subfolders to include', str, nargs='+') = None,
         skip: Param("subfolders to skip", str, nargs='+') = None):

    "generate combo dataset"
    if skip and only:
        print('you can skip subfolder or whitelist them but not both')
        return 1

    src_dirs = []
    for src in sources:
        sub_fldrs = subfolders(src) #subfolders normally e.g. IMM10ms
        if skip:  src_dirs += [fldr for fldr in sub_fldrs if fldr.stem not in skip]
        elif only: src_dirs += [fldr for fldr in sub_fldrs if fldr.stem in only]
        else: src_dirs += sub_fldrs

    tif_srcs = []

    for src in src_dirs:
        tif_srcs += build_tifs(src)        
        

    tif_src_df = pd.DataFrame(tif_srcs)
    
    tif_src_df[['category','dsplit','ftype','uint8','LRmean','LRsd','LRall_rmax','LRall_mi','LRall_ma','LRmi','LRma','LRrmax','LRx','LRy','HRmean','HRsd','HRall_rmax','HRall_mi','HRall_ma','HRmi','HRma','HRrmax','nc','c','HRx','HRy','LRfn','HRfn']].to_csv(out, header=True, index=False)
