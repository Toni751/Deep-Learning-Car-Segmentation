import numpy as np
import torch.optim as optim
import math
import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from torchmetrics.functional import dice
import torch.nn.functional as F
from unets import UNet, UNetPlusPlus

print("Started running car segmentation model.")
BATCH_SIZE = 32
ARRAYS_FOLDER = './arrays_rotated/'  # ARRAYS_FOLDER = 'carseg_data/arrays_rotated/'

image_data_list = []
target_list = []

npy_files = [f for f in os.listdir(ARRAYS_FOLDER) if f.endswith('.npy')]
for file in npy_files:
    file_path = os.path.join(ARRAYS_FOLDER, file)

    # Load the numpy array and normalize by dividing with the maximum value
    npy_file = np.load(file_path)

    image = torch.from_numpy(npy_file[:, :, 0:3]) / 255 # First 3 channels are the image data
    image = image.permute(2, 0, 1)  # Reshaping from HxWxC to CxHxW

    # Use this line to have a mask of the shape: HxW (Called class indices, I think this is the correct way to go)
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


def dice_coeff(outputs, targets):
    # Ensure that outputs is a float tensor and apply softmax
    outputs = F.softmax(outputs, dim=1).float()

    # Ensure that targets is a long tensor (class indices)
    targets = targets.long()

    # Compute dice coefficient
    return dice(outputs.argmax(dim=1), targets)


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
    if not torch.cuda.is_available():
        print("CUDA NOT AVAILABLE!!!!")
    else:
        print("CUDA WORKING!!!")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    train_steps = math.ceil(len(train_set) / BATCH_SIZE)
    train_losses = []
    val_losses = []
    train_dices = []
    val_accuracies = []

    for epoch in range(1, epochs + 1):
        epoch_loss = 0
        epoch_dice = 0
        model.train()

        for step, (inputs, masks) in enumerate(train_loader, 1):
            inputs, masks = inputs.to(device), masks.to(device)
            optimizer.zero_grad()
            output = model(inputs)

            batch_loss = loss_fn(output, masks)
            batch_loss.backward()
            optimizer.step()
            epoch_loss += batch_loss.item()

            batch_dice = dice_coeff(output, masks)
            epoch_dice += batch_dice

        epoch_dice /= train_steps
        train_dices.append(epoch_dice)
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


model = UNet()
model.apply(weights_init)
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0003)
loss_fn = nn.CrossEntropyLoss()  # this should also apply log-softmax to the output
save_path = 'model.pth'
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
train_model(model, 10, optimizer, loss_fn, save_path)