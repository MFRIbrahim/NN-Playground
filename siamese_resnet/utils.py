from albumentations.augmentations.transforms import JpegCompression, RandomBrightness
import torch
import torchvision
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import config as c


def save_model(model, optimizer, path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, path)

def load_model(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

def get_transforms():
    train_transform = A.Compose([
        A.Resize(height=c.IMAGE_HEIGHT, width=c.IMAGE_WIDTH),
        A.GaussNoise(p=1),
        # A.HorizontalFlip(p=0.5),
        # A.VerticalFlip(p=0.5),
        # A.Blur(p=0.5),
        # A.RandomGamma(p=0.5),
        # A.RandomBrightness(p=0.5),
        A.Normalize(
            mean=c.CHANNEL_MEANS,
            std=c.CHANNEL_STDS,
            max_pixel_value=255
        ),
        ToTensorV2(transpose_mask=True)
    ])

    val_transform = A.Compose([
        A.Resize(height=c.IMAGE_HEIGHT, width=c.IMAGE_WIDTH),
        A.Normalize(
            mean=c.CHANNEL_MEANS,
            std=c.CHANNEL_STDS,
            max_pixel_value=255
        ),
        ToTensorV2(transpose_mask=True)
    ])

    return train_transform, val_transform

def get_loader(dataset):
    loader = DataLoader(
        dataset,
        batch_size=c.VAL_BATCH_SIZE,
        shuffle=True,
        num_workers=c.N_WORKERS,
        pin_memory=c.PIN_MEMORY,
        drop_last=True
    )

    return loader

def get_error(model, loader):
    mean_error = 0
    
    model.eval()
    with torch.no_grad():
        for img_1, img_2, targets in loader:
            img_1 = img_1.to(c.DEVICE)
            img_2 = img_2.to(c.DEVICE)
            targets = targets.to(c.DEVICE)

            output = torch.sigmoid(model(img_1, img_2))
            output = (output > 0.5).float()

            difference = torch.abs(output - targets)
            mean_error += (torch.sum(difference) / difference.shape[0]).item()
    
    return mean_error / len(loader)
