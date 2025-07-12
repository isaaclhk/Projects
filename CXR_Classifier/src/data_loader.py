# imports
import os
import logging
import shutil

import kagglehub
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms

from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

    
class PneumoniaDataset(Dataset):
    '''
    A custom PyTorch Dataset for loading 
    chest X-ray images for pneumonia classification.

    The dataset expects a directory structure like:
        root_dir/
            NORMAL/
                img1.jpeg
                ...
            PNEUMONIA/
                person1_bacteria_1.jpeg
                person2_virus_2.jpeg
                ...

    Labels are mapped as follows:
        - 'normal'   -> 0
        - 'bacteria' -> 1
        - 'virus'    -> 2

    Parameters:
        data_dir (str): Path to the root directory containing the image folders.
        transform (callable, optional): Optional transform to be applied on a sample.
    '''

    def __init__(self, data_dir, transform=None):
        self.root_dir = data_dir
        self.transform = transform
        self.samples = []

        for label in ['NORMAL', 'PNEUMONIA']:
            class_dir = os.path.join(data_dir, label)
            for fname in os.listdir(class_dir):
                if fname.startswith('.'):
                    continue # exclude .ds_store
                fpath = os.path.join(class_dir, fname)
                if os.path.isfile(fpath):
                    if label == 'NORMAL':
                        self.samples.append((fpath, 'normal'))
                    else:
                        if 'bacteria' in fname.lower():
                            self.samples.append((fpath, 'bacteria'))
                        elif 'virus' in fname.lower():
                            self.samples.append((fpath, 'virus'))

        self.label_map = {'normal': 0, 'bacteria': 1, 'virus': 2}

    def __len__(self):
        '''
        Returns:
            int: Total number of samples in the dataset.
        '''
        return len(self.samples)

    def __getitem__(self, idx):
        '''
        Retrieves the image and label at the specified index.

        Parameters:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (image, label) where image is a transformed PIL Image and label is an integer.
        '''
        img_path, label_str = self.samples[idx]
        image = Image.open(img_path)
        label = self.label_map[label_str]

        if self.transform:
            image = self.transform(image)

        return image, label

def load_data(destination_dir: str) -> None:
    '''
    Downloads and moves the Chest X-ray Pneumonia dataset to a specified directory.
    Note:
        This combines the train and validation sets.
    '''
    def move_directory_contents(src_dir: str, dst_dir: str) -> None:
        '''
        Moves images from src_dir to the corresponding subfolder in dst_dir
        '''
        for subfolder in ['NORMAL', 'PNEUMONIA']:
            src_subdir = os.path.join(src_dir, subfolder)
            dst_subdir = os.path.join(dst_dir, subfolder)

            os.makedirs(dst_subdir, exist_ok=True)

            for item in os.listdir(src_subdir):
                src_path = os.path.join(src_subdir, item)
                dst_path = os.path.join(dst_subdir, item)
                shutil.move(src_path, dst_path)

    # Check if data already exists
    if os.path.exists(destination_dir) and os.listdir(destination_dir):
        logger.info(f"Dataset already exists at {destination_dir}. Skipping download.")
        return

    # Download latest version
    logger.info("Downloading data from Kaggle..")
    try:
        path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
        data_path = os.path.join(path, 'chest_xray', 'chest_xray')
        
        # Combine train and validation sets
        logger.debug("Combining train and val set..")
        try:
            data_path_train = os.path.join(data_path, 'train')
            data_path_val = os.path.join(data_path, 'val')
            move_directory_contents(data_path_val, data_path_train)
            os.rename(data_path_train, os.path.join(data_path, 'trainval'))
        except Exception as e:
            logger.error(f"Failed to combine train and val set: {e}")

        # Ensure destination directory exists
        os.makedirs(destination_dir, exist_ok=True)

        # Move dataset to destination
        shutil.move(os.path.join(data_path, 'trainval'), destination_dir)
        shutil.move(os.path.join(data_path, 'test'), destination_dir)
        logger.info(f'Data loaded to {destination_dir}.')

        # Clean up
        shutil.rmtree(data_path_val) 
        shutil.rmtree(path)
        
    except Exception as e:
        logger.error(f"Failed to load data: {e}")


def split_dataset(
    dataset: Dataset, val_ratio: float=0.2, seed:int=42
    ) -> tuple[Subset, Subset]:
    '''
    Splits a PyTorch dataset into stratified train and validation subsets.

    Parameters:
        dataset (Dataset): A PyTorch dataset.
        val_ratio (float): Proportion of the dataset to include in the validation split.
        seed (int): Random seed for reproducibility.

    Returns:
        train_subset (Subset): Stratified training subset.
        val_subset (Subset): Stratified validation subset.
    '''
    # Extract labels
    labels = [dataset[i][1] for i in range(len(dataset))]

    # Stratified split
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    train_idx, val_idx = next(splitter.split(X=labels, y=labels))

    # Create subsets
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)

    return train_subset, val_subset


def transform(augment: bool = True):
    '''
    Returns a set of image transformations for EfficientNet-V2-S.

    Parameters:
        augment (bool): If True, applies additional data augmentation suitable for training.
                        If False, applies only standard preprocessing for inference.

    Returns:
        torchvision.transforms.Compose: A composition of image transformations.

    When augment=True, the transformations include:
        - Resize the shorter side to 384 using bilinear interpolation
        - Center crop to 384x384
        - Apply random color jitter (brightness and contrast)
        - Convert to tensor with values in [0.0, 1.0]
        - Normalize using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]

    When augment=False:
        - Resize the shorter side to 384 using bilinear interpolation
        - Center crop to 384x384
        - Convert to tensor with values in [0.0, 1.0]
        - Normalize using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]
    '''
    if augment:
        efficientnet_v2_s_transform = transforms.Compose([
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.Resize(
                384, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(384),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),  # Converts to [0.0, 1.0] range
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        efficientnet_v2_s_transform = transforms.Compose([
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.Resize(
                384, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(384),
            transforms.ToTensor(),  # Converts to [0.0, 1.0] range
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    return efficientnet_v2_s_transform

def get_dataloader(
    dataset: Dataset, 
    shuffle: bool, 
    batch_size: int=8, 
    seed:int=42
    ):
    '''
    Creates a DataLoader from a given dataset with reproducibility.

    Args:
        dataset (Dataset): The dataset to load.
        shuffle (bool): Whether to shuffle the data.
        batch_size (int, optional): Number of samples per batch, defaults to 8.
        seed (int, optional): Random seed for reproducibility, defaults to 42.

    Returns:
        DataLoader: PyTorch DataLoader for the dataset.
    '''
    # rng for reproducibility
    generator = torch.Generator()
    generator.manual_seed(seed)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True,
        generator=generator
    )
    return dataloader