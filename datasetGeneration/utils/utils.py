"utility methods for generating movies from learners"
#from fastai import *
from fastai.vision import *
#from fastai.callbacks import *
import shutil
from skimage.io import imsave
import imageio

import PIL
import numpy as np
import cv2
from fastprogress import progress_bar
from pathlib import Path
import torch
import math

from time import sleep
from skimage.util import random_noise
from skimage import filters
from skimage.metrics import structural_similarity as ssim


__all__ = ['ensure_folder', 'subfolders','build_tile_info', 'generate_tiles',
           'draw_random_tile', 'draw_specific_tile', 'calc_image_deviation']

def ensure_folder(fldr, clean=False):
    fldr = Path(fldr)
    if fldr.exists() and clean:
        print(f'wiping {fldr.stem} in 5 seconds')
        sleep(5.)
        shutil.rmtree(fldr)
    if not fldr.exists(): fldr.mkdir(parents=True, mode=0o775, exist_ok=True)
    return fldr


def subfolders(p):
    return [sub for sub in p.iterdir() if sub.is_dir()]


def build_tile_info(data, tile_sz, train_samples, valid_samples, only_categories=None, skip_categories=None):
    if skip_categories == None: skip_categories = []
    if only_categories == None: only_categories = []
    if only_categories: skip_categories = [c for c in skip_categories if c not in only_categories]

    def get_category(p):
        return p.parts[-2]

    def get_mode(p):
        return p.parts[-3]

    def is_only(fn):
        return (not only_categories) or (get_category(fn) in only_categories)

    def is_skip(fn):
        return get_category(fn) in skip_categories

    def get_img_size(p):
        with PIL.Image.open(p) as img:
            h,w = img.size
        return h,w

    all_files = [fn for fn in list(data.glob('**/*.tif')) if is_only(fn) and not is_skip(fn)]
    img_sizes = {str(p):get_img_size(p) for p in progress_bar(all_files)}

    files_by_mode = {}

    for p in progress_bar(all_files):
        category = get_category(p)
        mode = get_mode(p)
        mode_list = files_by_mode.get(mode, {})
        cat_list = mode_list.get(category, [])
        cat_list.append(p)
        mode_list[category] = cat_list
        files_by_mode[mode] = mode_list

    def pull_random_tile_info(mode, tile_sz):
        files_by_cat = files_by_mode[mode]
        category=random.choice(list(files_by_cat.keys()))
        img_file=random.choice(files_by_cat[category])
        h,w = img_sizes[str(img_file)]
        return {'mode': mode,'category': category,'fn': img_file, 'tile_sz': tile_sz, 'h': h, 'w':w}


    tile_infos = []
    for i in range(train_samples):
        tile_infos.append(pull_random_tile_info('train', tile_sz))
    for i in range(valid_samples):
        tile_infos.append(pull_random_tile_info('valid', tile_sz))

    tile_df = pd.DataFrame(tile_infos)[['mode','category','tile_sz','h','w','fn']]
    return tile_df


def draw_tile(img, tile_sz):
    max_x,max_y = img.shape
    x = random.choice(range(max_x-tile_sz)) if max_x > tile_sz else 0
    y = random.choice(range(max_y-tile_sz)) if max_y > tile_sz else 0
    xs = slice(x,min(x+tile_sz, max_x))
    ys = slice(y,min(y+tile_sz, max_y))
    tile = img[xs,ys].copy()
    return tile, (xs,ys)
  
#def draw_specific_tile(img, tile_sz, x, y):
    #max_x,max_y = img.shape
    #x = 4*x if max_x > 4*tile_sz else 0
    #y = 4*y if max_y > 4*tile_sz else 0
    #xs = slice(x,min(x+4*tile_sz, max_x))
    #ys = slice(y,min(y+4*tile_sz, max_y))
    #tile = img[xs,ys].copy()
    #return tile, (xs,ys)
  
def draw_specific_tile(img, LRcrop, box, magnification):
    max_x,max_y = img.shape
    SSIM_index = {}
    PSNR_index = {}
    l = -10
    u = 10
    for row in range(l,u):
        for column in range(l,u):
            x = (magnification * box[0])+column
            y = (magnification * box[1])+row
            xend = (magnification *box[2])+column
            yend = (magnification *box[3])+row
            
            xs = slice(x,min(xend, max_x))
            ys = slice(y,min(yend, max_y))
            if yend <= max_y and xend < max_x:
                tile = img[xs,ys].copy()
                SSIM_index[(column, row)] = calc_image_deviation(tile,np.array(LRcrop))[1]
                PSNR_index[(column, row)] = calc_image_deviation(tile,np.array(LRcrop))[0]
    best = max(SSIM_index, key=SSIM_index.get)
    #if max(SSIM_index.values()) < 0.32:
        #print("PSNR")
        #best = max(PSNR_index, key=PSNR_index.get)
        #print(best)
        #print (max(PSNR_index.values()))
        
    #else:
        #print("SSIM")
        #best = max(SSIM_index, key=SSIM_index.get)
        #print(best)
        #print(max(SSIM_index.values()))
        
    column = best[0]
    row = best[1]
    boxEstDeviationXY = [column,row]
    x = (magnification * box[0])+column
    y = (magnification * box[1])+row
    xend = (magnification *box[2])+column
    yend = (magnification *box[3])+row
    xs = slice(x,min(xend, max_x))
    ys = slice(y,min(yend, max_y))
    tile = img[xs,ys].copy()
    deviation = calc_image_deviation(tile, np.array(LRcrop))
    SSIM = deviation[1]
    PSNR = deviation[0]
    return PIL.Image.fromarray(tile), SSIM , PSNR, boxEstDeviationXY
    

def calc_image_deviation(HR,LR):
    ##Image Data is np.array
    height, width = LR.shape
    downHR = cv2.resize(HR, dsize=(width, height), interpolation=cv2.INTER_CUBIC)
    PSNR = cv2.PSNR(downHR,LR)
    SSIM = ssim(downHR,LR)
    return PSNR, SSIM

def check_tile(img, thresh, thresh_pct):
    return (img > thresh).mean() > thresh_pct

def draw_random_tile(img_data, tile_sz, thresh, thresh_pct):
    max_tries = 200

    found_tile = False
    tries = 0
    while not found_tile:
        tile, (xs,ys) = draw_tile(img_data, tile_sz)
        found_tile = check_tile(tile, thresh, thresh_pct)
        # found_tile = True
        tries += 1
        if tries > (max_tries/2): thresh_pct /= 2
        if tries > max_tries: found_tile = True
    box = [xs.start, ys.start, xs.stop, ys.stop]
    return PIL.Image.fromarray(tile), box

def generate_tiles(dest_dir, tile_info, scale=4, crap_dirs=None, crap_func=None):
    tile_data = []
    dest_dir = ensure_folder(dest_dir)
    shutil.rmtree(dest_dir)
    if crap_dirs:
        for crap_dir in crap_dirs.values():
            if crap_dir:
                shutil.rmtree(crap_dir)

    last_fn = None
    tile_info = tile_info.sort_values('fn')
    for row_id, tile_stats in progress_bar(list(tile_info.iterrows())):
        mode = tile_stats['mode']
        fn = tile_stats['fn']
        tile_sz = tile_stats['tile_sz']
        category = tile_stats['category']
        if fn != last_fn:
            img = PIL.Image.open(fn)
            img_data = np.array(img)
            img_max = img_data.max()
            img_data /= img_max

            thresh = 0.01
            thresh_pct = (img_data.mean() > 1) * 0.5
            last_fn = fn
            tile_folder = ensure_folder(dest_dir/mode/category)
        if crap_dirs:
            crap_dir = crap_dirs[tile_sz]
            crap_tile_folder = ensure_folder(crap_dir/mode/category) if crap_dir else None
        else:
            crap_tile_folder = None
            crap_dir = None

        crop_img, box = draw_random_tile(img_data, tile_sz, thresh, thresh_pct)
        crop_img.save(tile_folder/f'{row_id:05d}_{fn.stem}.tif')
        if crap_func and crap_dir:
            crap_img = crap_func(crop_img, scale=scale)
            crap_img.save(crap_tile_folder/f'{row_id:05d}_{fn.stem}.tif')
        tile_data.append({'tile_id': row_id, 'category': category, 'mode': mode, 'tile_sz': tile_sz, 'box': box, 'fn': fn})
    pd.DataFrame(tile_data).to_csv(dest_dir/'tiles.csv', index=False)
