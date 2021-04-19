import torch
import torch.nn as nn
import torch.optim as optim
import config as c
from utils import (
    save_model,
    load_model,
    get_transforms,
    get_loader,
    get_error
)
from model import ResNet50
from dataset import ImageDataset

def train(model, loader, optimizer, loss_fn, scheduler):
    model.train()
    mean_loss = 0

    for img_1, img_2, targets in loader:
        img_1 = img_1.to(c.DEVICE)
        img_2 = img_2.to(c.DEVICE)
        targets = targets.float().to(c.DEVICE)

        # forward pass
        output = model(img_1, img_2)
        loss = loss_fn(output, targets)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # add to mean_loss
        mean_loss += loss.item()

    return mean_loss / len(loader)

def val(model, loader, loss_fn):
    model.eval()
    mean_loss = 0

    with torch.no_grad():
        for img_1, img_2, targets in loader:
            img_1 = img_1.to(c.DEVICE)
            img_2 = img_2.to(c.DEVICE)
            targets = targets.float().to(c.DEVICE)

            # forward pass
            output = model(img_1, img_2)
            loss = loss_fn(output, targets)

            # add to mean_loss
            mean_loss += loss.item()

    return mean_loss / len(loader)

def main():
    train_transform, val_transform = get_transforms()
    train_dataset = ImageDataset(c.TRAIN_IMG_DIR, train_transform)
    val_dataset = ImageDataset(c.VAL_IMG_DIR, val_transform)
    train_loader = get_loader(train_dataset)
    val_loader = get_loader(val_dataset)

    model = ResNet50(img_channels=3, n_output_features=4096).to(c.DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, 
                                                  base_lr=0.01,
                                                  max_lr=0.1)
    if c.LOAD_MODEL:
        load_model(model, optimizer, f'{c.MODEL_DIR}comparator_model.pth.tar')
    val_loss_tracker = float('inf')

    for epoch in range(c.N_EPOCHS):
        train_loss = train(model, train_loader, optimizer, loss_fn, scheduler)
        print(f"epoch: {epoch}")
        print(f"train_loss: {train_loss}")
        if (epoch+1)%10 == 0:
            val_loss = val(model, val_loader, loss_fn)
            print(f"val_loss: {val_loss}")
            if val_loss < val_loss_tracker:
                save_model(model, optimizer, f'{c.MODEL_DIR}comparator_model.pth.tar')
                val_loss_tracker = val_loss
            if val_loss > val_loss_tracker:
                print("Ending training.")
                break




if __name__ == '__main__':
    main()
