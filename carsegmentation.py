import numpy as np
import itertools
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, TensorDataset
import math
import os
import torchmetrics
import torch.nn.functional as F


print("Started running car segmentation model.")
BATCH_SIZE = 64
#ARRAYS_FOLDER = './arrays/'
ARRAYS_FOLDER = 'carseg_data/arrays_rotated/'
NUM_CLASSES = 9

image_data_list = []
target_list = []


def one_hot_mask(mask):
    out = np.zeros((mask.size, NUM_CLASSES), dtype=np.float_)
    out[np.arange(mask.size), mask.ravel()] = 1
    out.shape = mask.shape + (NUM_CLASSES,)
    return out


npy_files = [f for f in os.listdir(ARRAYS_FOLDER) if f.endswith('.npy')]
for file in npy_files:
    file_path = os.path.join(ARRAYS_FOLDER, file)
    
    # Load the numpy array and normalize by dividing with the maximum value
    npy_file = np.load(file_path)

    image = torch.from_numpy(npy_file[:, :, 0:3]) / 255 # First 3 channels are the image data
    image = image.permute(2, 0, 1)  # Reshaping from HxWxC to CxHxW

    # Use this two lines to have a mask of the shape: CxHxW (Called probabilities in the documentation)
    # target = one_hot_mask((npy_file[:, :, 3] % 90) // 10) # Last channel is the mask and we prepare it for one_hot
    # target = torch.from_numpy(target).permute(2, 0, 1) # Reshaping from HxWxC to CxHxW

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


class ConvBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Initialize weights using He initialization
        for layer in self.conv_block.children():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        return self.conv_block(x)




class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # output shape excluding channels (same for both height and width) is:
        # out = (in - 1) * stride - 2 * padding + (kernel_size - 1) + 1
        # here, with padding = 0, we get:
        # out = (stride * in) - (2 * stride) + kernel
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)  # x1 and x2 need to have the same number of rows, I think
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        # the CrossEntropyLoss automatically wraps this in a LogSoftmax, that's why we don't do it here
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.inc = ConvBlock(3, 64, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        # Additional layer
        self.down5 = Down(1024, 2048)

        self.up1 = Up(2048, 1024)
        self.up2 = Up(1024, 512)
        self.up3 = Up(512, 256)
        self.up4 = Up(256, 128)

        # Additional layer
        self.up5 = Up(128, 64)

        self.outc = OutConv(64, NUM_CLASSES)

    def forward(self, x):
        x1 = self.inc(x)  # x1 HxW: 256x256
        x2 = self.down1(x1)  # x2 HxW: 128x128
        x3 = self.down2(x2)  # x3 HxW: 64x64
        x4 = self.down3(x3)  # x4 HxW: 32x32
        x5 = self.down4(x4)  # x5 HxW: 16x16

        # Additional layer
        x6 = self.down5(x5)  # x6 HxW: 8x8

        x = self.up1(x6, x5)  # up(x6) gives 16x16, concat with x5, HxW remains 16x16 and the channels are added
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2)

        # Additional layer
        x = self.up5(x, x1)  # up(x4) gives 32x32, concat with x3, HxW remains 32x32 and the channels are added

        logits = self.outc(x)
        return logits


# TODO: maybe here we want to use the dice coefficient instead? (torchmetrics.Dice ?)
# TODO: this needs a fix
def accuracy(outputs, targets):
    return 1


def dice_coeff(outputs, targets):
    # Ensure that outputs is a float tensor and apply softmax
    outputs = F.softmax(outputs, dim=1).float()

    # Ensure that targets is a long tensor (class indices)
    targets = targets.long()

    # Compute dice coefficient
    dice = torchmetrics.functional.dice(outputs.argmax(dim=1), targets)
    return dice



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

        print(f"Epoch train loss: {train_losses[-1]}, train accuracy: {train_dices[-1]}")
        print(f"Epoch validation loss: {val_losses[-1]}, validation accuracy: {val_accuracies[-1]}")

    if save_path is not None:
        save_model(model, optimizer, save_path)

    test_loss, test_acc = evaluate_val_test_set(model, device, loss_fn, len(test_set), test_loader)
    print(f"Test loss: {test_loss}, test accuracy: {test_acc}")


#model = UNet()
#optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
#loss_fn = nn.CrossEntropyLoss()  # this should also apply log-softmax to the output
#save_path = 'model.pth'
#train_model(model, 10, optimizer, loss_fn, save_path)
def run_grid_search(param_grid, num_epochs=10):
    # Generate all possible combinations of hyperparameters
    all_params = [dict(zip(param_grid.keys(), values)) for values in itertools.product(*param_grid.values())]

    for params in all_params:
        print(f"\nTraining model with hyperparameters: {params}")

        model = UNet()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
        criterion = nn.CrossEntropyLoss()

        train_loader = DataLoader(train_set, batch_size=params['batch_size'], shuffle=True)
        val_loader = DataLoader(val_set, batch_size=params['batch_size'], shuffle=False)

        for epoch in range(num_epochs):
            model.train()  # Ensure the model is in training mode

            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)  # Move data to the same device as the model
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

        # Evaluate the model on the validation set
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        print(f'Validation Loss: {val_loss}')


param_grid = {
    'learning_rate': [1e-3, 1e-4, 1e-5],
    'weight_decay': [0, 1e-4, 1e-3],
    'batch_size': [16, 32, 64]
}

# Run grid search
run_grid_search(param_grid)

