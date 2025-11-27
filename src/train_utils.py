import torch
import numpy as np

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
    