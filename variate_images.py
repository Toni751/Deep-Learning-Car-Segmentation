import numpy as np
import os
from scipy.ndimage import rotate


ARRAYS_FOLDER = './arrays/'
ROTATED_FOLDER = './arrays_rotated/'


def rotate_images(image, target, angle):
    image = rotate(image, angle, reshape=False, order=0)
    target = rotate(target, angle, reshape=False, order=0)
    return image, target


def mirror_images(image, target):
    image = np.fliplr(image)
    target = np.fliplr(target)
    return image, target


def preprocess_file(files, shouldRotate=False, shouldMirror=False, angle=0):
    for file in files:
        new_file_name = ''
        file_path = os.path.join(ARRAYS_FOLDER, file)
        sample_tensor = np.load(file_path)

        original_image_data = sample_tensor[:, :, 0:3]  # First 3 channels are the image data
        augmented_image_data = original_image_data

        original_target = sample_tensor[:, :, 3]  # Fourth channel contains target values
        original_target = np.expand_dims(original_target, axis=-1)  # Add third dimension to the target
        augmented_target = original_target

        if shouldRotate and angle != 0:
            new_file_name += "angle_" + str(angle) + "_"
            augmented_image_data, augmented_target = rotate_images(augmented_image_data, augmented_target, angle)

        if shouldMirror:
            new_file_name += "mirrored_"
            augmented_image_data, augmented_target = mirror_images(augmented_image_data, augmented_target)

        # Save the augmented images and targets
        augmented_file_path = os.path.join(ROTATED_FOLDER, f"{new_file_name}{file}")
        np.save(augmented_file_path, np.concatenate([augmented_image_data, augmented_target], axis=-1))


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
