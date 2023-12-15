import numpy as np
import torch.optim as optim
import math
import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from torchmetrics.functional import dice
import torch.nn.functional as F
from torchvision.models.segmentation import fcn_resnet50
from torchvision.models.resnet import ResNet50_Weights
from unets import UNet, UNetPlusPlus
import matplotlib.pyplot as plt
import csv


BATCH_SIZE = 32
ARRAYS_FOLDER = './arrays_rotated/'
# ARRAYS_FOLDER = 'carseg_data/arrays_rotated/'

image_data_list = []
target_list = []

npy_files = [f for f in os.listdir(ARRAYS_FOLDER) if f.endswith('.npy')]
for file in npy_files:
    file_path = os.path.join(ARRAYS_FOLDER, file)
    npy_file = np.load(file_path)
    image = torch.from_numpy(npy_file[:, :, 0:3]) / 255
    image = image.permute(2, 0, 1)
    target = torch.as_tensor((npy_file[:, :, 3] % 90) // 10, dtype=torch.long)
    image_data_list.append(image)
    target_list.append(target)

images_tensor = torch.stack(image_data_list, dim=0)
masks_tensor = torch.stack(target_list, dim=0)
dataset = TensorDataset(images_tensor, masks_tensor)

DATASET_LENGTH = len(dataset)
train_size = math.floor(DATASET_LENGTH * 0.8)
val_size = math.floor(DATASET_LENGTH * 0.1)
test_size = DATASET_LENGTH - train_size - val_size

generator_seed = torch.Generator().manual_seed(0)
train_set, temp_set = random_split(dataset, [train_size, val_size + test_size], generator=generator_seed)
val_set, test_set = random_split(temp_set, [val_size, test_size], generator=generator_seed)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

r_map = {0: 0, 10: 250, 20: 19, 30: 249, 40: 10, 50: 149, 60: 5, 70: 20, 80: 249, 90: 0}
g_map = {0: 0, 10: 149, 20: 98, 30: 249, 40: 248, 50: 7, 60: 249, 70: 19, 80: 9, 90: 0}
b_map = {0: 0, 10: 10, 20: 19, 30: 10, 40: 250, 50: 149, 60: 9, 70: 249, 80: 250, 90: 0}

class_color_map = {
    0: (r_map[0], g_map[0], b_map[0]),
    1: (r_map[10], g_map[10], b_map[10]),
    2: (r_map[20], g_map[20], b_map[20]),
    3: (r_map[30], g_map[30], b_map[30]),
    4: (r_map[40], g_map[40], b_map[40]),
    5: (r_map[50], g_map[50], b_map[50]),
    6: (r_map[60], g_map[60], b_map[60]),
    7: (r_map[70], g_map[70], b_map[70]),
    8: (r_map[80], g_map[80], b_map[80]),
    9: (r_map[90], g_map[90], b_map[90]),
}


def save_loss_accuracy_to_file(train_losses, train_dices, val_losses, val_accuracies, class_color_map, file_path='dices_original.csv'):
    with open(file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        header = ['Epoch', 'Train Loss'] + [f'Train Dice Class {i} (Color: {class_color_map[i]})' for i in range(len(train_dices[0]))] + [
            'Validation Loss', 'Validation Accuracy']
        csv_writer.writerow(header)
        for epoch, train_loss, train_dices_epoch, val_loss, val_accuracy in zip(range(1, len(train_losses) + 1),
                                                                              train_losses, train_dices,
                                                                              val_losses,
                                                                              val_accuracies):
            row = [epoch, train_loss] + train_dices_epoch + [val_loss, val_accuracy]
            csv_writer.writerow(row)

def plot_loss_and_accuracy_from_file(file_path='loss_accuracy.csv'):
    with open(file_path, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)  # Skip the header row
        data = list(zip(*csv_reader))

        epochs, train_losses, train_dices, val_losses, val_accuracies = map(list, data)

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_losses, label='Train Loss')
        plt.plot(epochs, val_losses, label='Validation Loss')
        plt.title('Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_dices, label='Train Accuracy')
        plt.plot(epochs, val_accuracies, label='Validation Accuracy')
        plt.title('Accuracies')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.show()


def get_pretrained_model():
    pretrained_model = fcn_resnet50(weights_backbone=ResNet50_Weights.DEFAULT)
    pretrained_model.classifier[-1] = UNet(512)

    for p in pretrained_model.backbone.parameters():
        p.requires_grad = False

    return pretrained_model


def dice_coeff(outputs, targets):
    outputs = F.softmax(outputs, dim=1).float()
    targets = targets.long()
    return dice(outputs.argmax(dim=1), targets)


def dice_coeff_per_class(outputs, targets, num_classes):
    outputs = F.softmax(outputs, dim=1).float()
    targets = targets.long()

    dice_coeffs = torch.zeros(num_classes, dtype=torch.float)

    for i in range(num_classes):
        class_outputs = (outputs.argmax(dim=1) == i).float()
        class_targets = (targets == i).float()
        intersection = torch.sum(class_outputs * class_targets)
        union = torch.sum(class_outputs) + torch.sum(class_targets)
        dice_coeffs[i] = (2.0 * intersection) / (union + 1e-8)

    return dice_coeffs


def save_model(model, optimizer, save_path):
    state = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(state, save_path)
    print(f'Model saved at {save_path}')


def load_model(model, optimizer, load_path):
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f'Model loaded from {load_path}')
    return model, optimizer


def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def evaluate_val_test_set(model, device, loss_fn, set_length, loader):
    with torch.no_grad():
        model.eval()
        set_loss = 0
        set_dice = 0
        for inputs, masks in loader:
            inputs, masks = inputs.to(device), masks.to(device)
            output = model(inputs)
            set_loss += loss_fn(output, masks).item()

            batch_dice = dice_coeff(output, masks)
            set_dice += batch_dice

        set_dice /= len(loader)
        set_loss /= math.ceil(set_length / BATCH_SIZE)
        return set_loss, set_dice


def train_model(model, epochs, optimizer, loss_fn, save_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    train_steps = math.ceil(len(train_set) / BATCH_SIZE)
    train_losses = []
    val_losses = []
    train_dices = []
    val_accuracies = []

    for epoch in range(1, epochs + 1):
        epoch_loss = 0
        epoch_dice = torch.zeros(9, dtype=torch.float)
        model.train()

        for step, (inputs, masks) in enumerate(train_loader, 1):
            inputs, masks = inputs.to(device), masks.to(device)
            optimizer.zero_grad()
            output = model(inputs)

            batch_loss = loss_fn(output, masks)
            batch_loss.backward()
            optimizer.step()
            epoch_loss += batch_loss.item()

            batch_dice = dice_coeff_per_class(output, masks, 9)
            epoch_dice += batch_dice

        epoch_dice /= train_steps
        train_dices.append(epoch_dice.tolist())  # Convert to list for saving to CSV
        train_losses.append(epoch_loss / train_steps)

        val_loss, val_acc = evaluate_val_test_set(model, device, loss_fn, len(val_set), val_loader)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch}, train loss: {train_losses[-1]}, train accuracy: {train_dices[-1]}")
        print(f"Epoch {epoch}, validation loss: {val_losses[-1]}, validation accuracy: {val_accuracies[-1]}")

        scheduler.step()  # Step the learning rate scheduler

    if save_path is not None:
        save_model(model, optimizer, save_path)

    test_loss, test_acc = evaluate_val_test_set(model, device, loss_fn, len(test_set), test_loader)
    print(f"Test loss: {test_loss}, test accuracy: {test_acc}")
    save_loss_accuracy_to_file(train_losses, train_dices, val_losses, val_accuracies, class_color_map)


model = UNetPlusPlus()
model.apply(weights_init)
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.0003)
loss_fn = nn.CrossEntropyLoss()  # this should also apply log-softmax to the output
save_path = 'model.pth'
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
train_model(model, 14, optimizer, loss_fn, save_path)

# TO DO GRID SEARCH USE THE CODE BELOW

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# # Define the hyperparameters to search
# learning_rates = [1e-3, 5e-4, 1e-4]
# weight_decays = [3e-4, 5e-4]
# epochs_list = [12]
# batch_sizes = [32, 64]
#
# # Initialize variables to store the best hyperparameters and corresponding performance
# best_hyperparameters = None
# best_val_accuracy = 0.0
#
# # Iterate over all combinations of hyperparameters
# for learning_rate, weight_decay, epochs, batch_size in itertools.product(learning_rates, weight_decays, epochs_list,
#                                                                          batch_sizes):
#     print(f"Testing hyperparameters: lr={learning_rate}, wd={weight_decay}, epochs={epochs}, batch_size={batch_size}")
#
#     # Create a new model for each combination
#     model = UNetPlusPlus()
#     model.apply(weights_init)
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
#     loss_fn = nn.CrossEntropyLoss()
#     scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
#
#     # Train the model with the current hyperparameters
#     save_path = f'model_lr{learning_rate}_wd{weight_decay}_epochs{epochs}_batch{batch_size}.pth'
#     train_model(model, epochs, optimizer, loss_fn, save_path)
#
#     # Evaluate the model on the validation set
#     _, val_accuracy = evaluate_val_test_set(model, device, loss_fn, len(val_set), val_loader)
#
#     print(f"Validation accuracy: {val_accuracy}")
#
#     # Update the best hyperparameters if the current model performs better
#     if val_accuracy > best_val_accuracy:
#         best_val_accuracy = val_accuracy
#         best_hyperparameters = {
#             'learning_rate': learning_rate,
#             'weight_decay': weight_decay,
#             'epochs': epochs,
#             'batch_size': batch_size
#         }
#
# print("Grid search complete.")
# print("Best hyperparameters:", best_hyperparameters)
