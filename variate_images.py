import numpy as np
import os
from scipy.ndimage import rotate
from PIL import Image
import shutil

ARRAYS_FOLDER = './arrays/'
LANDSCAPES_FOLDER = './landscapes/'
ROTATED_FOLDER = './arrays_rotated/'
r_map = {0: 0, 10: 250, 20: 19, 30: 249, 40: 10, 50: 149, 60: 5, 70: 20, 80: 249, 90: 0}
g_map = {0: 0, 10: 149, 20: 98, 30: 249, 40: 248, 50: 7, 60: 249, 70: 19, 80: 9, 90: 0}
b_map = {0: 0, 10: 10, 20: 19, 30: 10, 40: 250, 50: 149, 60: 9, 70: 249, 80: 250, 90: 0}

allowed_mask_values = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
allowed_input_values = np.arange(256)


def rotate_images(image, target, angle):
    image = rotate(image, angle, reshape=False, order=0)
    target = rotate(target, angle, reshape=False, order=0)
    return image, target


def mirror_images(image, target):
    image = np.fliplr(image)
    target = np.fliplr(target)
    return image, target


def preprocess_file(files, shouldRotate=False, shouldMirror=False, shouldChangeBackground=False, angle=0):
    step = 0
    for file in files:
        new_file_name = ''
        file_path = os.path.join(ARRAYS_FOLDER, file)
        sample_tensor = np.load(file_path)

        original_image_data = sample_tensor[:, :, 0:3]  # First 3 channels are the image data
        augmented_image_data = original_image_data

        original_target = sample_tensor[:, :, 3]  # Fourth channel contains target values
        original_target = np.expand_dims(original_target, axis=-1)  # Add third dimension to the target
        augmented_target = original_target

        # CAN ONLY BE APPLIED TO BLACK IMAGES !!!!
        if shouldChangeBackground == True:
            new_file_name += "background_"
            # Determine the landscape image filename based on step count
            landscape_image_filename = f"{step + 1:04d}.jpg"
            landscape_image_path = os.path.join(LANDSCAPES_FOLDER, landscape_image_filename)

            # Replace black background with the specified landscape image
            augmented_image_data = replace_black_background(augmented_image_data, landscape_image_path)

        if shouldRotate == True and angle != 0:
            new_file_name += "angle_" + str(angle) + "_"
            augmented_image_data, augmented_target = rotate_images(augmented_image_data, augmented_target, angle)

        if shouldMirror == True:
            new_file_name += "mirrored_"
            augmented_image_data, augmented_target = mirror_images(augmented_image_data, augmented_target)

        # Save the augmented images and targets
        augmented_file_path = os.path.join(ROTATED_FOLDER, f"{new_file_name}{file}")
        np.save(augmented_file_path, np.concatenate([augmented_image_data, augmented_target], axis=-1))


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


def getRandomAngles():
    return np.random.choice([45, 90, 135, 180, 225, 270, 315])

def preprocess_input():
    black_files = [f for f in os.listdir(ARRAYS_FOLDER) if f.startswith('black_5') and f.endswith('.npy')]
    photo_files = [f for f in os.listdir(ARRAYS_FOLDER) if f.startswith('photo') and f.endswith('.npy')]
    orange_files = [f for f in os.listdir(ARRAYS_FOLDER) if f.startswith('orange') and f.endswith('.npy')]

    # The original images must always be saved here!!!

    # Preprocess photo images
    preprocess_file(photo_files, shouldRotate=True, angle=45)
    preprocess_file(photo_files, shouldRotate=True, angle=90)
    preprocess_file(photo_files, shouldRotate=True, angle=135)
    preprocess_file(photo_files, shouldRotate=True, angle=180)
    preprocess_file(photo_files, shouldRotate=True, angle=225)
    preprocess_file(photo_files, shouldRotate=True, angle=270)
    preprocess_file(photo_files, shouldRotate=True, angle=315)
    preprocess_file(photo_files, shouldRotate=True, angle=45, shouldMirror=True)
    preprocess_file(photo_files, shouldRotate=True, angle=90, shouldMirror=True)
    preprocess_file(photo_files, shouldRotate=True, angle=135, shouldMirror=True)
    preprocess_file(photo_files, shouldRotate=True, angle=180, shouldMirror=True)
    preprocess_file(photo_files, shouldRotate=True, angle=225, shouldMirror=True)
    preprocess_file(photo_files, shouldRotate=True, angle=270, shouldMirror=True)
    preprocess_file(photo_files, shouldRotate=True, angle=315, shouldMirror=True)
    preprocess_file(photo_files, shouldMirror=True)
    # Save original images
    for file in photo_files:
        original_file_path = os.path.join(ROTATED_FOLDER, f"original_{file}")
        file_path = os.path.join(ARRAYS_FOLDER, file)
        tensor = np.load(file_path)
        np.save(original_file_path, tensor)

    # Preprocess black images
    preprocess_file(black_files, shouldRotate=True, angle=getRandomAngles())
    # Save original images
    for file in black_files:
        original_file_path = os.path.join(ROTATED_FOLDER, f"original_{file}")
        file_path = os.path.join(ARRAYS_FOLDER, file)
        tensor = np.load(file_path)
        np.save(original_file_path, tensor)

    # Preprocess orange images
    for file in orange_files:
        original_file_path = os.path.join(ROTATED_FOLDER, f"original_{file}")
        file_path = os.path.join(ARRAYS_FOLDER, file)
        tensor = np.load(file_path)
        np.save(original_file_path, tensor)


# Create the rotated folder if it doesn't exist
if not os.path.exists(ROTATED_FOLDER):
    os.makedirs(ROTATED_FOLDER)

# Call the preprocess_input function
preprocess_input()
