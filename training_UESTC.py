import os
import glob
import argparse
import torch.utils
# import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from get_model import get_model
from dataset import UESTC_Dataset
import copy
import time
import math
# from generate_LinOSS_pytorch import create_pytorch_model
# from mamba_linoss import create_mamba_linoss_model


class WarmupLR(torch.optim.lr_scheduler._LRScheduler):
    """Warmup learning rate scheduler"""
    def __init__(self, optimizer, warmup_epochs, base_lr, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr
        super(WarmupLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            lr_scale = (self.last_epoch + 1) / self.warmup_epochs
            return [self.base_lr * lr_scale for _ in self.optimizer.param_groups]
        else:
            return [self.base_lr for _ in self.optimizer.param_groups]


def apply_gradient_clipping(model, max_norm=1.0):
    """Apply gradient clipping to prevent exploding gradients"""
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


def add_weight_noise(model, noise_std=1e-5):
    """Add small noise to weights for better generalization"""
    with torch.no_grad():
        for param in model.parameters():
            if param.requires_grad and param.dim() >= 2:  # Only for weight matrices
                noise = torch.randn_like(param) * noise_std
                param.add_(noise)


def train_model(model, dataloaders, criterion, optimizer, scheduler=None, num_epochs=25, patience=7):
    since = time.time()

    train_history = {}
    train_history['train_acc'] = []
    train_history['val_acc'] = []
    train_history['train_loss'] = []
    train_history['val_loss'] = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    # Early stopping variables
    no_improve_epochs = 0
    early_stop = False
    
    # Gradient clipping and noise settings
    use_grad_clipping = True
    use_weight_noise = False  # Enable for more regularization if needed
    max_grad_norm = 1.0

    for epoch in range(num_epochs):
        if early_stop:
            print(f"Early stopping triggered at epoch {epoch}")
            break
            
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode

            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        
                        # Apply gradient clipping
                        if use_grad_clipping:
                            apply_gradient_clipping(model, max_grad_norm)
                        
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # if phase == 'train':
            #     # train_history['train_acc'].append(epoch_acc)
            #     # train_history['train_loss'].append(epoch_loss)
            #     # Log to wandb
            #     wandb.log({"train_loss": epoch_loss, "train_acc": epoch_acc}, step=epoch)

            # deep copy the model
            if phase == 'val':
                # train_history['val_acc'].append(epoch_acc)
                # train_history['val_loss'].append(epoch_loss)
                # Log to wandb
                # wandb.log({"val_loss": epoch_loss, "val_acc": epoch_acc}, step=epoch)
                
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    # Reset early stopping counter when we find a better model
                    no_improve_epochs = 0
                else:
                    no_improve_epochs += 1
                    # if no_improve_epochs >= patience:
                    #     early_stop = True
                
                if scheduler is not None:
                    scheduler.step(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    
    # Log best validation accuracy to wandb
    # wandb.log({"best_val_acc": best_acc})

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, train_history



'''
DATA
'''
data_fold = "/hdd1/khanhnt/MICA-Net/MAMC/UESTC/inertial/"
modulation_dict = {}

data_train, target_train = [], []
data_val, target_val = [],[]
data_test, target_test = [],[]

class_counts_train = {i: 0 for i in range(32)}
class_counts_val = {i: 0 for i in range(32)}
class_counts_test = {i: 0 for i in range(32)}

for data_path in glob.glob(os.path.join(data_fold, "*", "*","*.csv")):
    label = int(data_path.split("/")[-2]) - 1
    modulation_dict[label] = (data_path.split("/")[-1]).split("_")[1]
    
    if data_path.split("/")[-3] == "train":
        data_train.append(data_path)
        target_train.append(label)
        class_counts_train[label] += 1
    
    elif data_path.split("/")[-3] == "val":
        data_val.append(data_path)
        target_val.append(label)
        class_counts_val[label] += 1
    
    elif data_path.split("/")[-3] == "test":
        data_test.append(data_path)
        target_test.append(label)
        class_counts_test[label] += 1


train_size = len(data_train)
val_size = len(data_val)
test_size = len(data_test)

modulation_list = []
for i in range(32):
    modulation_list.append(modulation_dict[i])


print(f"Total training samples: {train_size}")
print(f"Total validation samples: {val_size}")
print(f"Total test samples: {test_size}")

    
    # Create class distribution plot

train_dataset = UESTC_Dataset(data_train, target_train)
val_dataset = UESTC_Dataset(data_val, target_val)
test_dataset = UESTC_Dataset(data_test, target_test)


batch_size = 16
num_workers = 4
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)

dataloaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}
'''
MODEL
'''
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Initialize wandb
# wandb.init(
#     project="UESTC-Training",
#     config={
#         "learning_rate": 1e-3,
#         "weight_decay": 1e-5,
#         "epochs": 50,
#         "batch_size": 16,
#         "optimizer": "Adam",
#         "model": "backbone",
#         "scheduler": "ReduceLROnPlateau",
#     }
# )

arg = 0
# config = {
#     "d_model": 32,
#     "d_state": 32,
#     "d_conv": 4,
#     "expand": 6,
#     "n_layers": 2,
#     "num_classes": 32,
#     "length": 256
# }

# Enhanced config with optimized parameters
# config = {
#     "d_model": 256,  # Slightly increased from 32 for better capacity
#     "d_state": 128,
#     "d_conv": 4,
#     "expand": 2,
#     "n_layers": 2,  # Increased from 2 for deeper representation
#     "num_classes": 32,  # Your actual number of classes
#     "length": 256
# }
# backbone = LightweightMambaModel(config)
# backbone = get_model(arg)
# backbone = EnhancedLightweightMambaModel(config)
# print("Creating LinOSS Classification Model")
# model_cls = create_pytorch_model(
#     model_name='LinOSS',
#     data_dim=6,
#     label_dim=32,
#     hidden_dim=256,
#     num_blocks=1,
#     ssm_dim=256,
#     classification=True,
#     output_step=1,
#     linoss_discretization='IM',
#     norm_type='layer',
#     drop_rate=0.05,
#     device='cuda',
#     seed=42
# )

# config_info = {
#     'params': {
#         'd_model': 128, 'n_layers': 2, 'ssm_size': 32,
#         'a_init_method': 'log_uniform', 'a_init_range': (0.0, 2.0),
#         'dt_min': 0.001, 'dt_max': 0.1, 'bc_init_std': 0.01,
#         'log_eigenspectrum': True
#     }
# }

# model = create_mamba_linoss_model(**config_info['params'])
classifier = get_model(arg, model_type="momentum", momentum_beta=0.8, momentum_alpha=1.0).to(device)

params_to_update = classifier.parameters()
print("Params to learn:")
for name, param in classifier.named_parameters():
    if param.requires_grad:
        print(f"{name} will be updated")
    else:
        print(f"{name} will not be updated")


criterion = torch.nn.CrossEntropyLoss()

# Use lower weight decay with Adam
weight_decay = 1e-5
learning_rate = 1e-3
optimizer_ft = torch.optim.Adam(params_to_update, lr=learning_rate, weight_decay=weight_decay)

# Learning rate scheduler with patience tuned for better initialization
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode='max', factor=0.6, patience=5, verbose=True)

print("Warming up the model for 10 epochs")
# Warm up the model for 10 epochs
classifier, hist = train_model(classifier, dataloaders, criterion, optimizer_ft, scheduler, num_epochs=10, patience=7)

print("Training the model for 40 epochs")
model_ft, hist = train_model(classifier, dataloaders, criterion, optimizer_ft, scheduler, num_epochs=40, patience=7)



import numpy as np
running_corrects = 0
phase = 'test'

predictions = []
labelss = []
for inputs, labels in dataloaders[phase]:
    inputs = inputs.to(device)
    labels = labels.to(device)


    outputs = model_ft(inputs)
    loss = criterion(outputs, labels)
    _, preds = torch.max(outputs, 1)

    predictions.append(preds.cpu().detach().numpy())
    labelss.append(labels.data.cpu().detach().numpy())

    running_corrects += torch.sum(preds == labels.data)

epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

print(f"Test Accuracy: {epoch_acc}")

predictions = np.concatenate(predictions)
labelss = np.concatenate(labelss)

# from sklearn.metrics import confusion_matrix, classification_report
# from matplotlib import pyplot as plt
# import seaborn as sns

# def plot_confusion_matrix(y_test,y_scores, classNames):
#     # y_test=np.argmax(y_test, axis=1)
#     # y_scores=np.argmax(y_scores, axis=1)
#     classes = len(classNames)
#     cm = confusion_matrix(y_test, y_scores)
#     print("**** Confusion Matrix ****")
#     print(cm)
#     print("**** Classification Report ****")
#     print(classification_report(y_test, y_scores, target_names=classNames))
#     con = np.zeros((classes,classes))
#     for x in range(classes):
#         for y in range(classes):
#             con[x,y] = round(cm[x,y]/np.sum(cm[x,:]), 2)

#     plt.figure(figsize=(90,90))
#     sns.set(font_scale=4.5) # for label size
#     df = sns.heatmap(con, annot=True,fmt='.2', xticklabels= classNames , yticklabels= classNames)
#     df.figure.savefig("UESTC_cf_transformer.png")

# plot_confusion_matrix(labelss,predictions, modulation_list)

# wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(predictions, labelss, class_names=modulation_list)})

# wandb.log({"test_acc": epoch_acc})

# wandb.finish()