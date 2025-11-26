import os 
import torch 
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

import numpy as np
from sklearn.model_selection import train_test_split

SEED = 42

def get_basic_dataloader(
        data_dir: str,
        batch_size: int = 32,
):
    
    """
    Basic loader for the Plant Village Dataset 

    - loads all images from data_dir
    - applies a very simple transfrom (conver to tensor)
    - returns a single DataLoader and the list of class names
    """

    # Converts an image of shape (height, width, channel) with values [0,255]
    # into a Pytorch Tensor of shape (channel, height, width) with values [0,1]
    # transform = transforms.ToTensor()

    # Transform Image 
    transform = transforms.Compose([
        transforms.Resize((224,224)), # Resize all Images 
        transforms.ToTensor(),        # Convert to Tensor
        transforms.Normalize(         # Normalize to ImageNet 
            mean=[0.485, 0.456, 0.406], # mean of color channels (ResNet18)
            std=[0.229,0.224,0.225]     # std of color channels (ResNet18)
        )
    ])

    # Create an Image Folder Dataset
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Get the class names in order of their label indices (0,1,2,3 ...)
    class_names = dataset.classes

    return loader, class_names


class TransformSubset(Dataset):
    
    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices # Could be like [3,6,2,7]
        self.transform = transform

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        img, label  = self.dataset[actual_idx]

        if self.transform:
            img = self.transform(img)

        return img, label

def get_transforms():

    # We use two transforms (train, val_test) because we eventually want to
    # to augment the train data 
    train_transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.RandomResizedCrop(224,scale=(0.7,1.0)),
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(0.2),
        transforms.ColorJitter(brightness=0.15,
                                contrast=0.15,
                                saturation=0.15,
                                hue=0.02),

        transforms.ToTensor(),
        transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229,0.224,0.225]
        )
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229,0.224,0.225]
        )
    ])

    return train_transform, val_test_transform

def load_base_dataset(data_dir:str):
    return datasets.ImageFolder(root=data_dir)

def create_stratified_splits(indices, 
                  targets, 
                  val_split, 
                  test_split,
                  random_state=SEED):
        
    # Splitting the Train_Val and Test from the Full Dataset   
    train_val_idx, test_idx = train_test_split( 
        indices,
        test_size=test_split,
        stratify=targets,
        random_state=random_state
    )

    train_val_targets = [targets[i] for i in train_val_idx]

    # Splitting the Train and Val from the Train_Val Dataset
    train_idx, val_idx = train_test_split( 
        train_val_idx,
        test_size=val_split / (1-test_split),
        stratify=train_val_targets,
        random_state=random_state
    )

    return train_idx, val_idx, test_idx

def create_subsets(full_dataset,
                train_idx,
                val_idx,
                test_idx, 
                train_transform, val_test_transform):

    # Subsets share the same underlying Dataset so we 
    # cant directly do val_subset.dataset.transform = val_test_transform
    # so we use a custom wrapper class TransformSubset
    train_set = TransformSubset(full_dataset, train_idx, train_transform)
    val_set = TransformSubset(full_dataset, val_idx, val_test_transform)
    test_set = TransformSubset(full_dataset, test_idx, val_test_transform)

    return train_set, val_set, test_set

def create_data_loaders(train_dataset,val_dataset,test_dataset, 
                        batch_size:int = 32, num_workers = 0, random_state=SEED):
    
    generator = torch.Generator()
    generator.manual_seed(random_state)

    train_loader = None if not train_dataset else DataLoader(train_dataset, 
                                                             batch_size=batch_size,
                                                             shuffle=True,
                                                             num_workers = num_workers,
                                                             worker_init_fn=_seed_worker,
                                                             generator=generator)
    
    val_loader = None if not val_dataset else DataLoader(val_dataset,
                                                         batch_size=batch_size,
                                                         num_workers = num_workers,
                                                         worker_init_fn=_seed_worker,
                                                         shuffle=False)
    
    test_loader = None if not test_dataset else DataLoader(test_dataset,
                                                           batch_size=batch_size,
                                                           num_workers = num_workers,
                                                           worker_init_fn=_seed_worker,
                                                           shuffle=False)
    
    return train_loader, val_loader, test_loader

def get_datasets(
        data_dir:str,
        val_split: float = 0.1,
        test_split: float = 0.1,
):
    
    """
    Returns train, val and test dataloaders along with classes
    """
    
    # Transform the Data 
    train_transform, val_test_transform = get_transforms()

    # Load Full Dataset
    full_dataset = load_base_dataset(data_dir)
                                                                                
    targets = full_dataset.targets
    indices = list(range(len(full_dataset)))

    # Create Stratified Split
    train_idx, val_idx, test_idx = create_stratified_splits(indices,targets,val_split,test_split)

    # Create Datasets 
    train_set, val_set, test_set =  create_subsets(full_dataset,
                                                   train_idx, val_idx, test_idx, 
                                                   train_transform, val_test_transform)

    classes = full_dataset.classes

    return train_set, val_set, test_set, classes 

def _seed_worker(worker_id):

    """
    Sets the seed for each worker process to ensure reproducible augmentations
    """

    worker_seed = SEED + worker_id

    torch.manual_seed(worker_seed)
    np.random.seed(worker_seed)