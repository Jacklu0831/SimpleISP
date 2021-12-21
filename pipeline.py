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


def run_pipeline_v2(image_path, out_dir, demosaic_type='EA', save_as='jpg', save_dtype=np.uint8):
    # raw image data
    img = get_visible_raw_image(image_path)
    # metadata
    metadata = get_metadata(image_path)

    # normalization
    assert not 'linearization_table' not in metadata or metadata['linearization_table']
    img = normalize(img, metadata['black_level'], metadata['white_level'])
    save_image(img, image_path, osp.join(out_dir, 'normal'), save_as, save_dtype)

    # lens sharding correction
    gain_map_opcode = None
    if 'opcode_lists' in metadata:
        if 51009 in metadata['opcode_lists']:
            opcode_list_2 = metadata['opcode_lists'][51009]
            gain_map_opcode = opcode_list_2[9]
    if gain_map_opcode is not None:
        img = lens_shading_correction(img, gain_map_opcode=gain_map_opcode, bayer_pattern=metadata['cfa_pattern'])
    # save_image(img, image_path, osp.join(out_dir, 'lens_sharding_correction'), save_as, save_dtype)

    # white balance
    img = white_balance(img, metadata['as_shot_neutral'], metadata['cfa_pattern'])
    save_image(img, image_path, osp.join(out_dir, 'white_balance'), save_as, save_dtype)

    # demosaic
    img = demosaic(img, metadata['cfa_pattern'], output_channel_order='RGB', alg_type=demosaic_type)
    save_image(img, image_path, osp.join(out_dir, 'demosaic'), save_as, save_dtype)

    # xyz transformation
    img = apply_color_space_transform(img, metadata['color_matrix_1'], metadata['color_matrix_2'])
    save_image(img, image_path, osp.join(out_dir, 'xyz'), save_as, save_dtype)

    # rgb
    img = transform_xyz_to_srgb(img)
    img = fix_orientation(img, metadata['orientation'])
    save_image(img, image_path, osp.join(out_dir, 'srgb'), save_as, save_dtype)

    # gamma
    img = apply_gamma(img)
    save_image(img, image_path, osp.join(out_dir, 'gamma'), save_as, save_dtype)

    # tone map
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
