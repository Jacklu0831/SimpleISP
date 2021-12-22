import os.path as osp
from pathlib import Path
import glob
import os
import argparse

import cv2
import numpy as np
from pipeline_utils import get_visible_raw_image, get_metadata, normalize, white_balance, demosaic, \
    apply_color_space_transform, transform_xyz_to_srgb, apply_gamma, apply_tone_map, fix_orientation, \
    lens_shading_correction


def save_image(img, image_path, out_dir, save_as, save_dtype):
    os.makedirs(out_dir, exist_ok=True)

    stem = Path(image_path).stem
    max_val = 2 ** 16 if save_dtype == np.uint16 else 255
    img = (img[..., ::-1] * max_val).astype(save_dtype)

    out_path = osp.join(out_dir, stem) + f'.{save_as}'
    print('saving as', out_path)
    if save_as == 'jpg':
        cv2.imwrite(out_path, img, [cv2.IMWRITE_JPEG_QUALITY, 100])
    else:
        cv2.imwrite(out_path, img)


def color_code_img(img, cfa_pattern):
    color_img = np.zeros((*img.shape, 3))
    idx2by2 = [[0, 0], [0, 1], [1, 0], [1, 1]]
    step2 = 2
    for i, (idx, channel) in enumerate(zip(idx2by2, cfa_pattern)):
        color_img[:,:,channel][idx[0]::step2, idx[1]::step2] = img[idx[0]::step2, idx[1]::step2]
    color_img = color_img[:,:,::-1] / (np.max(color_img) - np.min(color_img))

    balanced_color_img = color_img.copy()
    balanced_color_img[:,:,1][1::step2, 1::step2] = 0.0
    return color_img, balanced_color_img


def run_pipeline_v2(image_path, out_dir, demosaic_type='EA', save_as='jpg', save_dtype=np.uint8):
    # raw: Adobe Digital Negative Raw Image file, need to be converted to srgb to fit the display specification
    # srgb: https://en.wikipedia.org/wiki/SRGB
    

    # raw image data
    img = get_visible_raw_image(image_path)
    metadata = get_metadata(image_path)
    save_image(img[::-1,:], image_path, osp.join(out_dir, 'raw'), save_as, save_dtype)

    # normalization: subtract by black level mask and normalize between [0, 1]
    # black level is the level of brightness at the darkest part of the image
    # white level is the level of brightness at the whitest part of the image
    # black_level: [0, 0, 0, 0]
    # white level: [1023]
    img = normalize(img, metadata['black_level'], metadata['white_level'])
    save_image(img[::-1,:], image_path, osp.join(out_dir, 'normal'), save_as, save_dtype)

    # lens sharding correction: don't talk about this for now, almost no impact
    # gain_map_opcode = None
    # if 'opcode_lists' in metadata:
    #     if 51009 in metadata['opcode_lists']:
    #         opcode_list_2 = metadata['opcode_lists'][51009]
    #         gain_map_opcode = opcode_list_2[9]
    # if gain_map_opcode is not None:
    #     img = lens_shading_correction(img, gain_map_opcode=gain_map_opcode, bayer_pattern=metadata['cfa_pattern'])
    # save_image(img, image_path, osp.join(out_dir, 'lens_sharding_correction'), save_as, save_dtype)

    # white balance: make image look more natural by adjusting image temperature
    # https://stephenstuff.wordpress.com/2014/01/05/an-introduction-to-digital-white-balance/
    # as_shot_neutral: [283/512, 1, 31/64] (each corresponds to rgb, divide each pixel by these then clip between 0, 1)
    # cfa_pattern (color filter pattern): 1021 GRBG
    # the as_shot_neutral here correspond appoximately to flueorescent (which makes sense cuz in room)
    img = white_balance(img, metadata['as_shot_neutral'], metadata['cfa_pattern'])
    save_image(img[::-1,:], image_path, osp.join(out_dir, 'white_balance'), save_as, save_dtype)

    # bayer pattern demonstration
    # https://www.google.com/search?q=GRBG+bayer+pattern&rlz=1C5CHFA_enCA850CA850&sxsrf=AOaemvLDnKJlZPZaLTIRFQW8xeKNNelb_Q:1640134350315&source=lnms&tbm=isch&sa=X&ved=2ahUKEwiu6dXKmPb0AhUJHzQIHQzxCrsQ_AUoAXoECAEQAw&biw=873&bih=998&dpr=2.2#imgrc=kieBJcRYliFAeM
    color_img, balanced_color_img = color_code_img(img, metadata['cfa_pattern'])
    save_image(color_img[::-1,::-1], image_path, osp.join(out_dir, 'color'), save_as, save_dtype)
    save_image(balanced_color_img[::-1,::-1], image_path, osp.join(out_dir, 'balanced_color'), save_as, save_dtype)

    # demosaic/debayer, edge aware interpolation
    # http://techtidings.blogspot.com/2012/01/demosaicing-exposed-normal-edge-aware.html
    img = demosaic(img, metadata['cfa_pattern'], output_channel_order='RGB', alg_type=demosaic_type)
    save_image(img[::-1,::-1], image_path, osp.join(out_dir, 'demosaic'), save_as, save_dtype)

    # CIE XYZ transformation: linear combination of rgb values based on color correction matrix
    # color space figure https://en.wikipedia.org/wiki/CIE_1931_color_space
    # https://www.imatest.com/docs/colormatrix/
    # color matrix 1: [85/128, -47/1024, -123/1024, -139/256, 1473/1024, 73/1024, -181/1024, 415/1024, 487/1024]
    img = apply_color_space_transform(img, metadata['color_matrix_1'])
    save_image(img[::-1,::-1], image_path, osp.join(out_dir, 'xyz'), save_as, save_dtype)

    # rgb
    # orientation: 3
    # xyz to rgb matrix from http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
    img = transform_xyz_to_srgb(img)
    img = fix_orientation(img, metadata['orientation'])
    save_image(img, image_path, osp.join(out_dir, 'srgb'), save_as, save_dtype)

    # gamma: (used to optimze bit usage, need to invert that)
    # gamma value of 1/2.2, standard for sRGB
    # https://en.wikipedia.org/wiki/Gamma_correction
    img = apply_gamma(img)
    save_image(img, image_path, osp.join(out_dir, 'gamma'), save_as, save_dtype)

    # tone map
    # global https://en.wikipedia.org/wiki/Tone_mapping
    img = apply_tone_map(img)
    save_image(img, image_path, osp.join(out_dir, 'tone'), save_as, save_dtype)


# output options: 'normal', 'white_balance', 'demosaic', 'xyz', 'srgb', 'gamma', 'tone'
parser = argparse.ArgumentParser(description='stuff')
parser.add_argument('--in-dir', default='/Users/jacklu/Documents/GitHub/SimpleISP/data/in/')
parser.add_argument('--out-dir', default='/Users/jacklu/Documents/GitHub/SimpleISP/data/out/')
args = parser.parse_args()

# processing a directory
images_dir = args.in_dir
image_paths = glob.glob(os.path.join(images_dir, '*.dng'))
print('found images:')
for p in image_paths:
    print('-', p)
print()

for image_path in image_paths:
    output_image = run_pipeline_v2(image_path, args.out_dir)
