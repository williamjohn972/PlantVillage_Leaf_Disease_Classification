import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

import pickle

from .train import Trainer, EarlyStopper
from .model import get_resnet18



def build_model(num_classes, dropout, device, freeze_base=True):
    
    model = get_resnet18(num_classes=num_classes, dropout=dropout)
    model.to(device)
    
    if freeze_base: 
        for params in model.parameters():           # Freezing Params belonging to all Layers
            params.requires_grad = False

        for params in model.fc.parameters():        # Enabling Params only for the final fc layer 
            params.requires_grad = True

    return model   

def build_optim(model, lr, weight_decay):
    
    trainable_params = filter(lambda param : param.requires_grad, model.parameters()) # Only Unfrozen Params
    optim = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    
    return optim

def build_trainer(model, loss_fn, optim,
                  early_stopper_patience, delta, save_checkpoints, checkpoint_path,
                  lr_factor, lr_patience, min_lr,
                  device):

    # LR Scheduler, Early Stopper
    early_stopper = EarlyStopper(patience=early_stopper_patience, delta=delta, 
                                 save_checkpoints=save_checkpoints, checkpoint_path=checkpoint_path, 
                                 mode="min",
                                 verbose=True)
    
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode="min", 
                                                              factor=lr_factor, patience=lr_patience,
                                                              min_lr=min_lr)
    # Trainer
    trainer = Trainer(model=model,
                      loss_fn=loss_fn, optim=optim,
                      early_stopper=early_stopper, lr_scheduler=lr_scheduler,
                      device=device)
    
    return trainer

def calc_class_weights(dataset, device):
    
    base_dataset_targets = np.array(dataset.dataset.targets, dtype=np.int64)
    indices = np.array(dataset.indices, np.int64)

    targets = base_dataset_targets[indices] 
    _, counts = np.unique(targets, return_counts=True)

    total_samples = targets.size

    class_frequencies = counts.astype(np.float32) / total_samples
    
    # Calculate weights and normalize
    epsilon = 1e-9 
    class_weights = 1.0 / (class_frequencies + epsilon)
    class_weights = class_weights / np.mean(class_weights)
    
    # 4. Convert to PyTorch Tensor
    return torch.tensor(class_weights, dtype=torch.float32).to(device)



def load_train_hist(file_path):
    with open(file_path, 'rb') as file:
        train_hist = pickle.load(file)

    return train_hist
    
def merge_train_hist(train_hist_1, train_hist_2):

    train_hist_merged = {}

    for key in train_hist_1.keys():

        # Check for metric lists
        if type(train_hist_1[key]) == list:

            train_hist_merged[key] = train_hist_1[key][:train_hist_1['best_epoch']] + train_hist_2[key]

    train_hist_merged["transition_epoch"] = train_hist_1['best_epoch'] + 1
    train_hist_merged["best_epoch"] = train_hist_2['best_epoch'] + train_hist_1['best_epoch']
    train_hist_merged["epochs"] = len(train_hist_merged["train_losses"])
    

    return train_hist_merged
    
def _plot_single_metric(ax, epochs_axis, 
                        train_data, val_data, 
                        ylabel,transition_epoch, best_epoch):

    """Plots a single metric (Loss or F1) on a given axis."""

    # Plot Training and Validation Curves
    ax.plot(epochs_axis, train_data, label='Train', color='darkblue', linewidth=3)
    ax.plot(epochs_axis, val_data, label='Validation', color='red', linewidth=3)

    # Add Vertical Line for Phase Transition (Unfreeze Point)
    ax.axvline(x=transition_epoch, color='orange', linestyle='-', linewidth=3, 
                label='Phase 2 Start (Unfreeze)', alpha=0.7)
                
    # Add Vertical Line for Final Best Checkpoint
    ax.axvline(x=best_epoch, color='green', linestyle='--', linewidth=3, 
                label=f'Best Checkpoint (Epoch {best_epoch})')

    ax.set_xlabel('Total Epochs')
    ax.set_ylabel(ylabel)
    ax.legend(loc='best', fontsize='small')
    ax.grid(axis='y', linestyle='--')


def plot_train_summary(train_history: dict, transition_epoch: int, save_path: str = None):
    """
    Plots Loss, Accuracy, and F1-Macro on separate figures sequentially.
    """
    
    # Prepare Data and Axis
    epochs_total = len(train_history.get('train_f1s', [])) 
    if epochs_total == 0:
        print("Error: History data is empty.")
        return
        
    epochs_axis = np.arange(1, epochs_total + 1)
    BEST_EPOCH = train_history.get('best_epoch', epochs_total) 
    
    # Define the metrics to plot
    metrics_to_plot = [
        ('Loss', 'train_losses', 'val_losses', 'Loss (Cross-Entropy)'),
        ('Accuracy', 'train_accs', 'val_accs', 'Accuracy (%)'),
        ('F1-Macro', 'train_f1s', 'val_f1s', 'F1-Macro Score')
    ]
    
    # Loop and Create a New Figure for Each Metric
    for metric_name, train_key, val_key, ylabel in metrics_to_plot:
        
        fig, ax = plt.subplots(1, 1, figsize=(20, 8))
        plt.style.use('fivethirtyeight')
        
        _plot_single_metric(
            ax, epochs_axis, 
            train_history[train_key], train_history[val_key], 
            ylabel=ylabel,
            transition_epoch=transition_epoch, 
            best_epoch=BEST_EPOCH
        )
        
        plt.suptitle(f'{metric_name}', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.show()

def plot_confusion_matrix(targets, predictions, class_names):

    cm = confusion_matrix(targets, predictions, normalize='true')
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

    plt.figure(figsize=(16,12))
    sns.heatmap(cm_df,
                annot=True,
                fmt='.3f',
                cmap="Reds",
                linewidths=0.5,
                linecolor="black")
    
    plt.title('Test Set Confusion Matrix', fontsize=16)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    plt.tight_layout()

