import numpy as np
import os

ARRAYS_FOLDER = './arrays/'
r_map = {0: 0, 10: 250, 20: 19, 30: 249, 40: 10, 50: 149, 60: 5, 70: 20, 80: 249, 90: 0}
g_map = {0: 0, 10: 149, 20: 98, 30: 249, 40: 248, 50: 7, 60: 249, 70: 19, 80: 9, 90: 0}
b_map = {0: 0, 10: 10, 20: 19, 30: 10, 40: 250, 50: 149, 60: 9, 70: 249, 80: 250, 90: 0}


def map_r_pixel(current_pixel):
    return r_map[current_pixel]


def map_g_pixel(current_pixel):
    return g_map[current_pixel]


def map_b_pixel(current_pixel):
    return b_map[current_pixel]


map_r_pixel_vec = np.vectorize(map_r_pixel, otypes=['uint8'])
map_g_pixel_vec = np.vectorize(map_g_pixel, otypes=['uint8'])
map_b_pixel_vec = np.vectorize(map_b_pixel, otypes=['uint8'])


def map_npy_mask(original_mask):
    mapped_r = map_r_pixel_vec(original_mask)
    mapped_g = map_g_pixel_vec(original_mask)
    mapped_b = map_b_pixel_vec(original_mask)
    output = np.stack([mapped_r, mapped_g, mapped_b], axis=2)
    return output


def preprocess_input():
    npy_files = [f for f in os.listdir(ARRAYS_FOLDER) if f.endswith('.npy')]

    step = 0
    for file in npy_files:
        step += 1
        if step % 100 == 0:
            print(step)

        file_path = os.path.join(ARRAYS_FOLDER, file)
        sample_tensor = np.load(file_path)

        image_data = sample_tensor[:, :, 0:3]  # First 3 channels are the image data
        target = map_npy_mask(sample_tensor[:, :, 3])  # Fourth channel contains target values
        np.save("./arrays_preprocessed/" + file, np.concatenate([image_data, target], axis=-1))
