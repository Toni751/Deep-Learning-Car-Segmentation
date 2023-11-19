import numpy as np
import os
from scipy.ndimage import rotate
from PIL import Image

ARRAYS_FOLDER = 'carseg_data/arrays/'
LANDSCAPES_FOLDER = 'carseg_data/images/landscapes/'
ROTATED_FOLDER = 'carseg_data/arrays_rotated/'
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


def replace_black_background(image_data, background_image_path):
    # Load the background image
    background_image = Image.open(background_image_path)

    # Resize the background image to match the size of the input image
    background_image = background_image.resize((image_data.shape[1], image_data.shape[0]))

    # Convert the background image to a NumPy array
    background_array = np.array(background_image)

    # Create a mask for black pixels in the input image
    black_pixels = (image_data[:, :, 0] == 0) & (image_data[:, :, 1] == 0) & (image_data[:, :, 2] == 0)

    # Replace RGB values for black pixels
    image_data[black_pixels, :3] = background_array[black_pixels, :3]

    return image_data


def preprocess_input():
    npy_files = [f for f in os.listdir(ARRAYS_FOLDER) if f.startswith('black_5') and f.endswith('.npy')]
    photo_files = [f for f in os.listdir(ARRAYS_FOLDER) if f.startswith('photo') and f.endswith('.npy')]
    orange_files = [f for f in os.listdir(ARRAYS_FOLDER) if f.startswith('orange') and f.endswith('.npy')]

    step = 0
    for file in npy_files:
        step += 1
        if step % 100 == 0:
            print(step)

        file_path = os.path.join(ARRAYS_FOLDER, file)
        sample_tensor = np.load(file_path)

        image_data = sample_tensor[:, :, 0:3]  # First 3 channels are the image data
        target = map_npy_mask(sample_tensor[:, :, 3])  # Fourth channel contains target values

        # Determine the landscape image filename based on step count
        landscape_image_filename = f"{step:04d}.jpg"
        landscape_image_path = os.path.join(LANDSCAPES_FOLDER, landscape_image_filename)

        # Replace black background with the specified landscape image
        image_data = replace_black_background(image_data, landscape_image_path)

        # Rotate the image and target at a random angle between 0 and 360 degrees
        random_angle = np.random.randint(0, 360)
        rotated_image_data = rotate(image_data, random_angle, reshape=False)
        rotated_target = rotate(target, random_angle, reshape=False)

        # Save the original and rotated images and targets
        original_file_path = os.path.join(ROTATED_FOLDER, f"original_{file}")
        rotated_file_path = os.path.join(ROTATED_FOLDER, f"rotated_{file}")

        np.save(original_file_path, np.concatenate([image_data, target], axis=-1))
        np.save(rotated_file_path, np.concatenate([rotated_image_data, rotated_target], axis=-1))

        # Delete the original file
        os.remove(file_path)

    for file in photo_files:
        step += 1
        if step % 100 == 0:
            print(step)

        file_path = os.path.join(ARRAYS_FOLDER, file)
        sample_tensor = np.load(file_path)

        image_data = sample_tensor[:, :, 0:3]  # First 3 channels are the image data
        target = map_npy_mask(sample_tensor[:, :, 3])  # Fourth channel contains target values

        # Rotate the image and target at a random angle between 0 and 360 degrees
        random_angle = np.random.randint(0, 360)
        rotated_image_data = rotate(image_data, random_angle, reshape=False)
        rotated_target = rotate(target, random_angle, reshape=False)

        # Save the original and rotated images and targets
        original_file_path = os.path.join(ROTATED_FOLDER, f"original_{file}")
        rotated_file_path = os.path.join(ROTATED_FOLDER, f"rotated_{file}")


        np.save(rotated_file_path, np.concatenate([rotated_image_data, rotated_target], axis=-1))

        # Delete the original file
        os.remove(file_path)

    for file in orange_files:
        step += 1
        if step % 100 == 0:
            print(step)

        file_path = os.path.join(ARRAYS_FOLDER, file)
        sample_tensor = np.load(file_path)

        image_data = sample_tensor[:, :, 0:3]  # First 3 channels are the image data
        target = map_npy_mask(sample_tensor[:, :, 3])  # Fourth channel contains target values

        # Rotate the image and target at a random angle between 0 and 360 degrees
        random_angle = np.random.randint(0, 360)
        rotated_image_data = rotate(image_data, random_angle, reshape=False)
        rotated_target = rotate(target, random_angle, reshape=False)

        # Save the original and rotated images and targets
        original_file_path = os.path.join(ROTATED_FOLDER, f"original_{file}")
        rotated_file_path = os.path.join(ROTATED_FOLDER, f"rotated_{file}")


        np.save(rotated_file_path, np.concatenate([rotated_image_data, rotated_target], axis=-1))

        # Delete the original file
        os.remove(file_path)


# Create the rotated folder if it doesn't exist
if not os.path.exists(ROTATED_FOLDER):
    os.makedirs(ROTATED_FOLDER)

# Call the preprocess_input function
preprocess_input()
