import glob
import os
import cv2
import numpy as np

from pipeline import run_pipeline


params = {
    'output_stage': 'tone',  # options: 'normal', 'white_balance', 'demosaic', 'xyz', 'srgb', 'gamma', 'tone'
    'save_as': 'png',  # options: 'jpg', 'png', 'tif', etc.
    'demosaic_type': 'menon2007',
    'save_dtype': np.uint16
}

# processing a directory
images_dir = '../data/'
image_paths = glob.glob(os.path.join(images_dir, '*.dng'))
for image_path in image_paths:
    output_image = run_pipeline(image_path, params)
    output_image_path = image_path.replace('.dng', '_{}.'.format(params['output_stage']) + params['save_as'])
    max_val = 2 ** 16 if params['save_dtype'] == np.uint16 else 255
    output_image = (output_image[..., ::-1] * 255).astype(params['save_dtype'])
    if params['save_as'] == 'jpg':
        cv2.imwrite(output_image_path, output_image, [cv2.IMWRITE_JPEG_QUALITY, 100])
    else:
        cv2.imwrite(output_image_path, output_image)
