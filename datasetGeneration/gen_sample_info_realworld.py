"""
gen_sample_info.py
-------
PSSR PIPELINE - STEP 1:
Understand your datasource - Extract metadata of images in the datasource(s).

This script is the first step of PSSR pipeline. It walks through files in your
datasources and extract the metadata of each frame/slice in all desired images,
which will serve as a guide map for training data generation. It allows users to
include images from multiple datasources, and each datasource folder can have
multiple subfolders, each of which includes images of a subcategory. Images of
each subcategory need to be split into folder 'train' and 'valid' beforehand.

The datasource folder has to follow the hierarchy as below:
- datasources
  |- live
  |  |- subcategory1
  |  |  |- train
  |  |  |   |- img1
  |  |  |   |- img2
  |  |  |   |- ...
  |  |  |- valid
  |  |      |- img1
  |  |      |- img2
  |  |      |- ...
  |  |
  |  |- subcategory2
  |  |  |- train
  |  |  |   |- img1
  |  |  |   |- img2
  |  |  |   |- ...
  |  |  |- valid
  |  |      |- img1
  |  |      |- img2
  |  |      |- ...
  |  |- ...
  |
  |- fixed
     |- subcategory1
     |  |- train
     |  |   |- img1
     |  |   |- img2
     |  |   |- ...
     |  |- valid
     |      |- img1
     |      |- img2
     |      |- ...
     |
     |- subcategory2
     |  |- train
     |  |   |- img1
     |  |   |- img2
     |  |   |- ...
     |  |- valid
     |      |- img1
     |      |- img2
     |      |- ...
     |- ...

Notes:
-------
1. Except folders named 'fixed' or 'live' (which refers to live cell or fixed samples),
'train' or 'valid', all other folders and files can be changed to different names accrodingly.
2. tif/tiff and czi, the two most widely used scientific image formats are
supported. Here are some additional important information:
- czi images: 3D (XYZ) and 4D (XYZT) stacks are acceptable. For multi-channel
              images, only the first channel will be included (editable in the script).
- tif/tiff images: 2D (XY) or 3D (XYZ/XYT) stacks are acceptable. Hyperstack images are
                   recommended to be preprocessed in ImageJ/FIJI.

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
  - 'multi': boolean, equals to 1 if any dimension of the source file other than
             X and Y larger than 1, which can be Z or T, in other words, if the
             source file of this slice is a z-stack or a time-lapse.
  - 'ftype': str, source file type, 'tif' or 'czi'.
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
  - 'nz': int, number of slices in Z dimension of the source image stack.
  - 'c': int, channel number of this frame. The first channel starts counting at 0.
  - 'nt': int, number of frames in T dimension of the source image stack.
  - 'z': int, depth number of this frame. The first slice starts counting at 0.
  - 't': int, time frame of this frame. The first frame starts counting at 0.
  - 'x': int, dimension in X.
  - 'y': int, dimension in Y.
  - 'fn': str, relative path of the source image.

Examples:
-------
Following are a couple of examples showing how to generate the metadata
from the datasource in different ways. Given the datasource folder is strctured
as below:
- datasources
  |- live
  |  |- mitotracker
  |  |  |- train
  |  |  |   |- mito_train1.tif
  |  |  |   |- mito_train2.tif
  |  |  |   |- ...
  |  |  |- valid
  |  |      |- mito_valid1.tif
  |  |      |- mito_valid2.tif
  |  |      |- ...
  |  |
  |  |- microtubules
  |     |- train
  |     |   |- microtubules_train1.tif
  |     |   |- microtubules_train2.tif
  |     |   |- ...
  |     |- valid
  |         |- microtubules_valid1.tif
  |         |- microtubules_valid2.tif
  |         |- ...
  |
  |- fixed
     |- neurons
     |  |- train
     |  |   |- neurons_train1.tif
     |  |   |- neurons_train2.tif
     |  |   |- ...
     |  |- valid
     |      |- neurons_valid1.tif
     |      |- neurons_valid2.tif
     |      |- ...
     |
     |- microtubules
        |- train
        |   |- microtubules_fixed_train1.tif
        |   |- microtubules_fixed_train2.tif
        |   |- ...
        |- valid
            |- microtubules_fixed_valid1.tif
            |- microtubules_fixed_valid2.tif
            |- ...

Example 1:
Only 'mitotracker' folder in datasource 'live' is needed. Name output
.csv file as 'live_mitotracker.csv'
python gen_sample_info.py --only mitotracker --out live_mitotracker.csv datasources/live

Example 2:
All subcategories in datasource 'live' are needed for training. Name
output file as 'live.csv'.
python gen_sample_info.py --out live.csv datasources/live

Example 3:
All subcategroies in datasource 'fixed' are needed for training except
files in 'microtubules'. Name output file as 'live.csv'
python gen_sample_info.py --skip microtubules --out live.csv datasources/fixed

Example 4:
Everything in folder 'datasources' are needed. Name output .csv file as 'all.csv'
python gen_sample_info.py --out all.csv datasources
"""
import yaml
from fastai.script import *
from fastai.vision import *
from utils import *
from pathlib import Path
from fastprogress import master_bar, progress_bar
from time import sleep
import shutil
import PIL
import czifile

PIL.Image.MAX_IMAGE_PIXELS = 99999999999999

def process_czi(item, category, mode):
#This function only takes the first channel of the czi files
#since those are the only mitotracker channels
    tif_srcs = []
    base_name = item.stem
    with czifile.CziFile(item) as czi_f:
        data = czi_f.asarray()
        axes, shape = get_czi_shape_info(czi_f)
        channels = shape['C']
        depths = shape['Z']
        times = shape['T']
        #times = min(times, 30) #ONLY USE FIRST 30 frames
        x,y = shape['X'], shape['Y']

        mid_depth = depths // 2
        depth_range = range(max(0,mid_depth-2), min(depths, mid_depth+2))
        is_multi = (times > 1) or (depths > 1)

        data = czi_f.asarray()
        all_rmax = data.max()
        all_mi, all_ma = np.percentile(data, [2,99.99])

        dtype = data.dtype
        #for channel in range(channels): #if other channels are needed, use this line
        for channel in range(0,1):
            for z in depth_range:
                for t in range(times):
                    idx = build_index(
                        axes, {
                            'T': t,
                            'C': channel,
                            'Z': z,
                            'X': slice(0, x),
                            'Y': slice(0, y)
                        })
                    img = data[idx]
                    mi, ma = np.percentile(img, [2,99.99])
                    if dtype == np.uint8: rmax = 255.
                    else: rmax = img.max()
                    tif_srcs.append({'fn': item, 'ftype': 'czi', 'multi':int(is_multi), 'category': category, 'dsplit': mode,
                                     'uint8': dtype == np.uint8, 'mi': mi, 'ma': ma, 'rmax': rmax,
                                     'all_rmax': all_rmax, 'all_mi': all_mi, 'all_ma': all_ma,
                                     'mean': img.mean(), 'sd': img.std(),
                                     'nc': channels, 'nz': depths, 'nt': times,
                                     'z': z, 't': t, 'c':channel, 'x': x, 'y': y})
    return tif_srcs

def is_live(item):
    return item.parent.parts[-4] == 'live'

def process_tif(item, category, mode):
    #item is tuple (hr,lr)
    hrItem = item[0]
    lrItem = item[1]
    
    tif_srcs = []
    #hr
    hrImg = PIL.Image.open(hrItem)
    hr_n_frames = hrImg.n_frames
    h_x,h_y = hrImg.size
    hr_is_multi = hr_n_frames > 1
    
    #lr
    lrImg = PIL.Image.open(lrItem)
    lr_n_frames = lrImg.n_frames
    l_x,l_y = lrImg.size
    lr_is_multi = lr_n_frames > 1
    #n_frames = min(n_frames, 30) #ONLY USE FIRST 30 frames


    #This is only done for HR data------------
    HRdata = []
    for n in range(hr_n_frames):
        hrImg.seek(n)
        hrImg.load()
        HRimg_data = np.array(hrImg)
        HRdata.append(HRimg_data)

    HRdata = np.stack(HRdata)
    HRall_rmax = HRdata.max()
    HRall_mi, HRall_ma = np.percentile(HRdata, [2,99.99])
    
    #This is only done for LR data------------
    LRdata = []
    for n in range(lr_n_frames):
        lrImg.seek(n)
        lrImg.load()
        LRimg_data = np.array(lrImg)
        LRdata.append(LRimg_data)

    LRdata = np.stack(LRdata)
    LRall_rmax = LRdata.max()
    LRall_mi, LRall_ma = np.percentile(LRdata, [2,99.99])


    #HR
    for n in range(hr_n_frames):
        HRimg_data = HRdata[n]
        HRdtype = HRimg_data.dtype
        HRmi, HRma = np.percentile(HRimg_data, [2,99.99])
        if HRdtype == np.uint8: HRrmax = 255.
        else: HRrmax = HRimg_data.max()
        if is_live(item[0]):
            t, z = n, 0
            nt, nz = hr_n_frames, 1
        else:
            HRt, HRz = 0, n
            HRnt, HRnz = 1, hr_n_frames
            
            
    #LR
    for n in range(lr_n_frames):
        LRimg_data = LRdata[n]
        LRdtype = LRimg_data.dtype
        LRmi, LRma = np.percentile(LRimg_data, [2,99.99])
        if LRdtype == np.uint8: LRrmax = 255.
        else: LRrmax = LRimg_data.max()
        if is_live(item[1]):
            t, z = n, 0
            nt, nz = lr_n_frames, 1
        else:
            LRt, LRz = 0, n
            LRnt, LRnz = 1, hr_n_frames
    #-------------------------------------------------
    
        tif_srcs.append({'HRfn': hrItem, 'LRfn': lrItem, 'ftype': 'tif', 'multi':int(hr_is_multi), 'category': category, 'dsplit': mode,
                         'uint8': HRdtype==np.uint8, 'HRmi': HRmi, 'HRma': HRma, 'HRrmax': HRrmax,
                         'HRall_rmax': HRall_rmax, 'HRall_mi': HRall_mi, 'HRall_ma': HRall_ma,
                         'HRmean': HRimg_data.mean(), 'HRsd': HRimg_data.std(),
                         'nc': 1, 'HRnz': HRnz, 'HRnt': HRnt,
                         'HRz': HRz, 'HRt': HRt, 'c':0, 'HRx': h_x, 'HRy': h_y,
                         'LRmi': LRmi, 'LRma': LRma, 'LRrmax': LRrmax,
                         'LRall_rmax': LRall_rmax, 'LRall_mi': LRall_mi, 'LRall_ma': LRall_ma,
                         'LRmean': LRimg_data.mean(), 'LRsd': LRimg_data.std(),
                         'LRnz': LRnz, 'LRnt': LRnt,
                         'LRz': LRz, 'LRt': LRt, 'LRx': l_x, 'LRy': l_y})
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
                '.tiff': process_tif,
                '.czi': process_czi,
            }
            
            map_f = item_map.get(item[0].suffix, process_unk)# item.suffix is ending of items filepath (e.g. .tif) ... item is a tuple containing hr & lr
            return map_f(item, category, mode) #map_f is either process_tif or process_czi
    except Exception as ex:
        print(f'err procesing: {item}')
        print(ex)
        return []

def build_tifs(src, mbar=None):
    tif_srcs = []
    for mode in ['train', 'valid', 'test']:
        live = src.parent.parts[-1] == 'live' #True or False
        hr_src_dir = src / "hr" / mode
        lr_src_dir = src / "lr" / mode
        category = src.stem #returns final path component without suffix, so without .tif ---> means filename or folder name --> Dataset Name
        hr_items = list(hr_src_dir.iterdir()) if hr_src_dir.exists() else []
        lr_items = list(lr_src_dir.iterdir()) if lr_src_dir.exists() else []
        items = list(zip(hr_items, lr_items)) if len(hr_items) == len(lr_items) and lr_items != [] and hr_items != [] else []
        #items is a list of tuples with the paths of hr and lr images (hr,lr)
        # Make sure, hr and lr are also the right pairs!!!
        if items:
            for p in progress_bar(items, parent=mbar):
                mbar.child.comment = mode
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

    mbar = master_bar(src_dirs)
    tif_srcs = []
    
    for src in mbar:
        mbar.write(f'process {src.stem}')        
        #lr_tif_srcs += build_tifs(src/"lr", mbar=mbar)
        tif_srcs += build_tifs(src, mbar=mbar)
        

    tif_src_df = pd.DataFrame(tif_srcs)
    
    tif_src_df[['category','dsplit','multi','ftype','uint8','LRmean','LRsd','LRall_rmax','LRall_mi','LRall_ma','LRmi','LRma','LRrmax','LRnz','LRnt','LRz','LRt','LRx','LRy','HRmean','HRsd','HRall_rmax','HRall_mi','HRall_ma','HRmi','HRma','HRrmax','HRnz','HRnt','nc','c','HRz','HRt','HRx','HRy','LRfn','HRfn']].to_csv(out, header=True, index=False)
