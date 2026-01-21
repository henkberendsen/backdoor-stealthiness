"""
Backdoor Attack Stealthiness Evaluation Utilities

This module provides a comprehensive suite of tools for evaluating the stealthiness
of backdoor attacks against image classification models. It includes:

- Custom dataset classes for various backdoor attack implementations
- Stealthiness metrics across three domains: input-space, feature-space, and parameter-space
- Data loading and model management utilities
- Feature extraction and visualization tools
- Evaluation and benchmarking functions

Supported Attacks:
    - BadNets, Blend, WaNet, BppAttack
    - Adaptive Patch/Blend
    - DFST, DFBA
    - Narcissus, Grond

Supported Metrics:
    - Input-space: L1/L2/Linf, MSE, PSNR, SSIM, LPIPS, IS, pHash, SAM
    - Feature-space: Silhouette Score (SS), Class-specific Davies-Bouldin Index (CDBI),
                     Discriminant Sliced-Wasserstein Distance (DSWD)
    - Parameter-space: Upper bound of Channel Lipschitzness Constant (UCLC),
                       Trigger-Activated Change (TAC), TAC-UCLC Product (TUP)

Author: Research Team
Reference: "Quiet Triggers, Loud Footprints: On Backdoor Attacks Stealthiness in Image Domain"
"""

import copy
import numpy as np
import os
import pandas as pd
import sys
import torch
import torchvision
import torchvision.transforms.v2 as T

from itertools import product
from matplotlib import pyplot as plt
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm

# Scikit-learn imports for metrics
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score, mean_squared_error
from sklearn.preprocessing import normalize

# Image quality metrics
from numpy.linalg import norm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import lpips
from torch.nn.functional import softmax
from scipy.special import rel_entr
from imagehash import phash
from torchmetrics.functional.image import spectral_angle_mapper
from ot import wasserstein_1d


# =============================================================================
# SECTION 1: CUSTOM DATASET CLASSES
# =============================================================================


class Imagenette(torchvision.datasets.VisionDataset):
    """
    Custom torchvision implementation of the Imagenette dataset.
    
    Unlike the official torchvision implementation, this class provides a 'data'
    attribute containing numpy arrays of all images, enabling consistent handling
    with CIFAR-10 and CIFAR-100 datasets.
    
    Attributes:
        classes (list): List of class names (10 classes)
        data (numpy.ndarray): Array of shape (N, H, W, 3) containing all images
        targets (list): List of integer labels for each image
        
    Args:
        root (str): Root directory containing 'train' and 'val' subdirectories
        train (bool): If True, loads training set; otherwise loads validation set
        transform (callable, optional): Transform to apply to images
        target_transform (callable, optional): Transform to apply to targets
        
    Example:
        >>> dataset = Imagenette(root='./data/imagenette2-160', train=True)
        >>> img, label = dataset[0]
        >>> print(f"Image shape: {img.shape}, Label: {dataset.classes[label]}")
    """
    
    def __init__(self, root=None, train=True, transform=None, target_transform=None):
        super().__init__(root, transform=transform, target_transform=target_transform)

        split = "train" if train else "val"
        img_folder = torchvision.datasets.ImageFolder(os.path.join(self.root, split))

        self.classes = [
            'tench', 'English springer', 'cassette player', 'chain saw', 'church',
            'French horn', 'garbage truck', 'gas pump', 'golf ball', 'parachute'
        ]
        self.data = []
        self.targets = []

        for img_path, label in img_folder.samples:
            img = np.array(Image.open(img_path).convert("RGB"))

            if len(img.shape) != 3:
                print(f"Warning: Invalid image shape at {img_path}")
                
            self.data.append(img)
            self.targets.append(label)

        self.data = np.stack(self.data)

    def __getitem__(self, index):
        """
        Retrieves image and label at specified index.
        
        Args:
            index (int): Index of the sample to retrieve
            
        Returns:
            tuple: (image, target) where image is a PIL Image or tensor (after transform)
        """
        img, target = self.data[index], self.targets[index]

        # Convert to PIL Image for consistency with other datasets
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
    
    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.data)


class BackdoorDataset(Dataset):
    """
    Base class for backdoor datasets with poisoned samples.
    
    This class wraps a standard dataset and adds backdoor-specific metadata,
    including poison indicators, original labels, and target class information.
    
    Attributes:
        bd_dataset: The underlying backdoored dataset
        classes (list): Class names
        target_class (int): Target class for backdoor attack
        original_labels (numpy.ndarray): Original (clean) labels before poisoning
        poison_lookup (numpy.ndarray): Boolean array indicating poisoned samples
        cross_lookup (numpy.ndarray): Boolean array indicating cross-trigger samples
        poisoned_labels (numpy.ndarray): Labels after poisoning (poisoned samples → target_class)
        
    Args:
        bd_dataset: Dataset containing backdoored images
        classes (list): List of class names
        target_class (int): Target class that poisoned samples should predict
        original_labels (numpy.ndarray): Array of original labels
        poison_lookup (numpy.ndarray): Boolean mask for poisoned samples
        cross_lookup (numpy.ndarray, optional): Boolean mask for cross-trigger samples
        
    Note:
        Cross-trigger samples are samples that have triggers but maintain their
        original labels (used in some adaptive attack strategies).
        
    Example:
        >>> dataset = BackdoorDataset(
        ...     bd_dataset=poisoned_dataset,
        ...     classes=['airplane', 'car', ...],
        ...     target_class=0,
        ...     original_labels=np.array([1, 2, 3, ...]),
        ...     poison_lookup=np.array([False, True, False, ...])
        ... )
    """
    
    def __init__(self, bd_dataset, classes, target_class, original_labels, 
                 poison_lookup, cross_lookup=None):
        self.bd_dataset = bd_dataset
        self.classes = classes
        self.target_class = target_class
        self.original_labels = original_labels
        self.poison_lookup = poison_lookup
        self.cross_lookup = cross_lookup if cross_lookup is not None else np.full(len(original_labels), False)

        # Create poisoned labels: copy originals and set poisoned indices to target class
        self.poisoned_labels = self.original_labels.copy()
        self.poisoned_labels[self.poison_lookup] = self.target_class

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.bd_dataset)
    
    def __getitem__(self, index):
        """
        Retrieves sample at specified index.
        
        Args:
            index (int): Index of sample to retrieve
            
        Returns:
            tuple: (image, label) pair from underlying dataset
        """
        return self.bd_dataset.__getitem__(index)


class BackdoorBenchDataset(BackdoorDataset):
    """
    Wrapper for BackdoorBench framework datasets.
    
    BackdoorBench is a comprehensive benchmark for backdoor attacks. This class
    adapts BackdoorBench datasets to work with the evaluation framework by
    extracting metadata and optionally replacing transforms.
    
    Args:
        bb_dataset: BackdoorBench dataset object (utils.bd_dataset_v2.dataset_wrapper_with_transform)
        target_class (int): Target class for the backdoor attack
        replace_transform (callable, optional): New transform to replace existing one
        
    Note:
        BackdoorBench datasets use poison_indicator values:
        - 0: Clean sample
        - 1: Poisoned sample (label changed to target)
        - 2: Cross-trigger sample (has trigger, original label)
        
    Example:
        >>> from backdoorbench.utils.save_load_attack import load_attack_result
        >>> result = load_attack_result('path/to/attack_result.pt')
        >>> dataset = BackdoorBenchDataset(
        ...     bb_dataset=result['bd_train'],
        ...     target_class=0,
        ...     replace_transform=my_transform
        ... )
    """
    
    def __init__(self, bb_dataset, target_class, replace_transform=None):
        classes = bb_dataset.wrapped_dataset.dataset.classes
        original_labels = np.array(bb_dataset.wrapped_dataset.dataset.targets)
        poison_lookup = bb_dataset.wrapped_dataset.poison_indicator == 1
        cross_lookup = bb_dataset.wrapped_dataset.poison_indicator == 2

        if replace_transform:
            bb_dataset.wrap_img_transform = replace_transform

        super().__init__(bb_dataset, classes, target_class, original_labels, 
                        poison_lookup, cross_lookup)

    def __getitem__(self, index):
        """
        Retrieves sample from BackdoorBench dataset.
        
        BackdoorBench returns multiple values; this method extracts only
        the image and backdoor label for compatibility.
        
        Args:
            index (int): Index of sample to retrieve
            
        Returns:
            tuple: (image, backdoor_label)
        """
        img, bd_label, dataset_idx, poison_bit, clean_label = self.bd_dataset.__getitem__(index)
        return img, bd_label


class AdapDataset(BackdoorDataset):
    """
    Dataset class for Adaptive Patch/Blend attacks.
    
    Adaptive attacks (Adap-Patch, Adap-Blend) save poisoned images as individual
    files. This class loads these images and integrates them into the clean dataset.
    
    Args:
        bd_record_path (str): Path to attack record directory containing:
            - data/train/ or data/test/: Poisoned image files
            - poison_indices: Indices of poisoned samples
            - cover_indices: Indices of cross-trigger samples
        target_class (int): Target class for backdoor attack
        split (str): Either 'train' or 'test'
        clean_dataset: Clean dataset to merge poisoned images into
        
    Directory Structure:
        bd_record_path/
        ├── data/
        │   ├── train/
        │   │   ├── 0.png
        │   │   ├── 1.png
        │   │   └── ...
        │   └── test/
        │       └── ...
        ├── poison_indices (torch tensor)
        └── cover_indices (torch tensor)
        
    Example:
        >>> clean_dataset = torchvision.datasets.CIFAR10(root='./data', train=True)
        >>> adap_dataset = AdapDataset(
        ...     bd_record_path='./record/adaptive_patch_resnet18_cifar10_p0.05',
        ...     target_class=0,
        ...     split='train',
        ...     clean_dataset=clean_dataset
        ... )
    """
    
    def __init__(self, bd_record_path, target_class, split, clean_dataset):
        bd_dataset_path = os.path.join(bd_record_path, "data", split)
        bd_dataset = copy.deepcopy(clean_dataset)

        classes = clean_dataset.classes
        original_labels = np.array(clean_dataset.targets)
        n_samples = len(clean_dataset)

        if split == "train":
            poison_indices = torch.load(os.path.join(bd_record_path, "poison_indices"))
            cover_indices = torch.load(os.path.join(bd_record_path, "cover_indices"))
        else:
            # Test set: all samples are poisoned
            poison_indices = range(n_samples)
            cover_indices = []

        # Replace benign images with poisoned versions at specified indices
        for i in np.concatenate([poison_indices, cover_indices]):
            i = int(i)
            img = Image.open(os.path.join(bd_dataset_path, f"{i}.png"))
            bd_dataset.data[i] = np.asarray(img)

            # Update label to target class for poisoned (not cross) samples
            if i in poison_indices:
                bd_dataset.targets[i] = target_class

        poison_lookup = np.array([i in poison_indices for i in range(n_samples)])
        cross_lookup = np.array([i in cover_indices for i in range(n_samples)])

        super().__init__(bd_dataset, classes, target_class, original_labels, 
                        poison_lookup, cross_lookup)


class DFSTDataset(BackdoorDataset):
    """
    Dataset class for DFST (Deep Feature Space Trojan) attacks.
    
    DFST generates poisoned images by optimizing triggers in the feature space.
    Poisoned images are saved in a poison_data.pt file. This class loads and
    integrates them into the clean dataset.
    
    Args:
        bd_record_path (str): Path to directory containing poison_data.pt
        target_class (int): Target class for backdoor attack
        split (str): Either 'train' or 'test'
        clean_dataset: Clean dataset to merge poisoned images into
        
    File Structure:
        poison_data.pt contains:
        - 'train': Tensor of poisoned training images (samples not of target class)
        - 'test': Tensor of poisoned test images
        - 'train_indices': Indices of poisoned samples in filtered trainset
        
    Note:
        DFST does not poison samples already belonging to the target class.
        For training, indices refer to the filtered dataset (excluding target class).
        
    Example:
        >>> clean_dataset = torchvision.datasets.CIFAR10(root='./data', train=True)
        >>> dfst_dataset = DFSTDataset(
        ...     bd_record_path='./record/dfst_resnet18_cifar10_p0.05',
        ...     target_class=0,
        ...     split='train',
        ...     clean_dataset=clean_dataset
        ... )
    """
    
    def __init__(self, bd_record_path, target_class, split, clean_dataset):
        classes = clean_dataset.classes
        original_labels = np.array(clean_dataset.targets)
        n_samples = len(clean_dataset)
        bd_dataset = copy.deepcopy(clean_dataset)
        bd_dataset_path = os.path.join(bd_record_path, "poison_data.pt")

        # Load poisoned images for non-target-class samples
        poison_data = torch.load(bd_dataset_path, weights_only=False)
        non_target_poisoned = poison_data[split]

        if split == "train":
            # Indices in the filtered trainset (excluding target class)
            poison_indices = poison_data["train_indices"]
            # Map to indices in full trainset
            true_indices = np.arange(n_samples)[original_labels != target_class].squeeze()
        else:
            # Test set: all samples are poisoned
            poison_indices = range(n_samples)
            true_indices = poison_indices

        # Initialize lookup arrays
        poison_lookup = np.full(n_samples, False)
        cross_lookup = np.full(n_samples, False)

        # Replace benign images with poisoned versions
        for i, poison_idx in enumerate(poison_indices):
            true_idx = true_indices[poison_idx]
            poison_lookup[true_idx] = True

            # Convert tensor to numpy array and update dataset
            poisoned_img = non_target_poisoned[i]
            bd_dataset.data[true_idx] = (poisoned_img * 255).permute(1, 2, 0).numpy()
            bd_dataset.targets[true_idx] = target_class

        super().__init__(bd_dataset, classes, target_class, original_labels, 
                        poison_lookup, cross_lookup)


class DFBADataset(BackdoorDataset):
    """
    Dataset class for DFBA (Data-Free Backdoor Attack).
    
    DFBA is a data-free attack that doesn't require training data. It applies
    a learned trigger (mask + perturbation) to test images.
    
    Args:
        clean_testset: Clean test dataset
        target_class (int): Target class for backdoor attack
        delta (torch.Tensor): Perturbation pattern (trigger)
        mask (torch.Tensor): Binary or continuous mask for trigger placement
        
    Trigger Application:
        backdoored_image = clean_image * (1 - mask) + delta * 255 * mask
        
    Note:
        - DFBA is a clean-label attack (doesn't change training labels)
        - All test samples are treated as poisoned
        - No training set is available for DFBA
        
    Example:
        >>> delta = torch.load('path/to/delta.pth')
        >>> mask = torch.load('path/to/mask.pth')
        >>> clean_testset = torchvision.datasets.CIFAR10(root='./data', train=False)
        >>> dfba_dataset = DFBADataset(
        ...     clean_testset=clean_testset,
        ...     target_class=0,
        ...     delta=delta,
        ...     mask=mask
        ... )
    """
    
    def __init__(self, clean_testset, target_class, delta, mask):
        classes = clean_testset.classes
        original_labels = np.array(clean_testset.targets)
        n_samples = len(clean_testset)
        poison_lookup = np.full(n_samples, True)  # All samples are poisoned
        cross_lookup = np.full(n_samples, False)
        bd_testset = copy.deepcopy(clean_testset)

        # Apply trigger to all test images
        for i in range(n_samples):
            # Rearrange dimensions: (H, W, C) -> (C, H, W)
            img = bd_testset.data[i].transpose(2, 0, 1)

            # Blend image and trigger using mask
            bd_img = img * (1 - mask) + (delta * 255) * mask

            # Restore original dimension order
            bd_testset.data[i] = bd_img.transpose(1, 2, 0)

            # Set label to target class
            bd_testset.targets[i] = target_class

        super().__init__(bd_testset, classes, target_class, original_labels, 
                        poison_lookup, cross_lookup)


class GrondDataset(BackdoorDataset):
    """
    Dataset class for Grond attacks.
    
    Grond is a clean-label backdoor attack. This class wraps the Grond
    implementation's POI and POI_TEST classes.
    
    Args:
        dataset (str): Dataset name ('cifar10', 'cifar100', etc.)
        transform (callable): Transform to apply to images
        target_class (int): Target class for backdoor attack
        record_path (str): Path to Grond attack record directory containing:
            - poison_indices.pth: Indices of poisoned training samples
            - Other Grond-specific files (trigger parameters, etc.)
        train (bool): If True, loads training set; otherwise loads test set
        
    Note:
        Requires Grond implementation to be available in ./grond/
        The grond directory must be added to sys.path before importing.
        
    Example:
        >>> import sys
        >>> sys.path.append('./grond')
        >>> from grond.poison_loader import POI, POI_TEST
        >>> 
        >>> grond_dataset = GrondDataset(
        ...     dataset='cifar10',
        ...     transform=transforms.ToTensor(),
        ...     target_class=0,
        ...     record_path='./record/grond_resnet18_cifar10_p0.007',
        ...     train=True
        ... )
    """
    
    def __init__(self, dataset, transform, target_class, record_path, train=True):
        # Import Grond's poison loader (must be in sys.path)
        grond_dir = os.path.abspath("./grond")
        if grond_dir not in sys.path:
            sys.path.append(grond_dir)
        from grond.poison_loader import POI, POI_TEST

        # Reconstruct poisoned dataset used in attack
        if train:
            poison_indices = torch.load(os.path.join(record_path, "poison_indices.pth"))
            data_dir = os.path.join("data", dataset)  # Adjust if needed
            poi_dataset = POI(
                dataset, 
                root=data_dir,
                poison_rate=None,  # Unused when poison_indices is provided
                transform=transform,
                poison_indices=poison_indices,
                target_cls=target_class,
                upgd_path=record_path
            )
        else:
            data_dir = os.path.join("data", dataset)
            poi_dataset = POI_TEST(
                dataset,
                root=data_dir,
                transform=transform,
                exclude_target=True,
                target_cls=target_class,
                upgd_path=record_path
            )

        # Extract metadata for BackdoorDataset
        classes = poi_dataset.cleanset.classes
        original_labels = np.array(poi_dataset.targets)
        n_samples = len(poi_dataset)
        poison_lookup = np.array([i in poison_indices for i in range(n_samples)]) if train else np.full(n_samples, True)
        cross_lookup = np.full(n_samples, False)

        super().__init__(poi_dataset, classes, target_class, original_labels, 
                        poison_lookup, cross_lookup)


# =============================================================================
# SECTION 2: UTILITY FUNCTIONS
# =============================================================================


def filter_target_class(dataset, target_class):
    """
    Removes all samples of the target class from a dataset.
    
    This is used to create test sets that exclude the target class, matching
    the typical evaluation setup for backdoor attacks.
    
    Args:
        dataset: PyTorch dataset with 'data' and 'targets' attributes
        target_class (int): Class to filter out
        
    Returns:
        Dataset with target class samples removed
        
    Modifies:
        - dataset.data: Filtered array of images
        - dataset.targets: Filtered list of labels
        
    Example:
        >>> testset = torchvision.datasets.CIFAR10(root='./data', train=False)
        >>> testset_filtered = filter_target_class(testset, target_class=0)
        >>> print(f"Original: {len(testset)}, Filtered: {len(testset_filtered)}")
    """
    targets_ndarray = np.array(dataset.targets)
    non_target_class = targets_ndarray != target_class
    dataset.data = dataset.data[non_target_class]
    dataset.targets = list(np.array(dataset.targets)[non_target_class])
    return dataset


def extract_preds(model, dataset, batch_size=100, device=None):
    """
    Extracts model predictions for all samples in a dataset.
    
    Args:
        model (torch.nn.Module): Neural network model
        dataset: PyTorch dataset
        batch_size (int, optional): Batch size for inference. Default: 100
        device (torch.device, optional): Device to run inference on. Default: auto-detect
        
    Returns:
        torch.Tensor: 1D tensor of predicted class indices (shape: [N])
        
    Example:
        >>> model = ResNet18(num_classes=10)
        >>> dataset = torchvision.datasets.CIFAR10(root='./data', train=False)
        >>> predictions = extract_preds(model, dataset)
        >>> accuracy = (predictions == torch.tensor(dataset.targets)).float().mean()
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    predictions = torch.tensor([])
    dl = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    for in_batch, _ in iter(dl):
        with torch.no_grad():
            pred_batch = model.forward(in_batch.to(device, non_blocking=True))
            predictions = torch.cat([predictions, pred_batch.cpu().argmax(dim=1)])

    return predictions


def extract_preds_and_features(model, dataset, penultimate=True, batch_size=100, device=None):
    """
    Extracts both predictions and intermediate layer features from a model.
    
    This function is useful for feature-space analysis of backdoor attacks,
    such as computing t-SNE embeddings or Wasserstein distances.
    
    Args:
        model (torch.nn.Module): Neural network with feature extraction capability
        dataset: PyTorch dataset
        penultimate (bool, optional): If True, extract penultimate layer features;
                                     otherwise extract features from layer used for TAC metric.
                                     Default: True
        batch_size (int, optional): Batch size for inference. Default: 100
        device (torch.device, optional): Device to run inference on. Default: auto-detect
        
    Returns:
        tuple: (predictions, features) where
            - predictions: 1D tensor of predicted classes (shape: [N])
            - features: 2D tensor of features (shape: [N, feature_dim])
            
    Note:
        Model must support one of these interfaces:
        - penultimate=True: model.forward(x, return_features=True) or model.forward_all_features(x)
        - penultimate=False: model.forward_all_features(x)
        
    Example:
        >>> model = ResNet18(num_classes=10)
        >>> dataset = torchvision.datasets.CIFAR10(root='./data', train=False)
        >>> preds, features = extract_preds_and_features(model, dataset)
        >>> print(f"Features shape: {features.shape}")  # e.g., [10000, 512]
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    predictions = torch.tensor([])
    features = torch.tensor([])
    dl = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    for in_batch, _ in iter(dl):
        size = len(in_batch)  # May be smaller than batch_size for final batch

        with torch.no_grad():
            if penultimate:
                # Extract features from penultimate layer
                pred_batch, feat_batch = model.forward(in_batch.to(device, non_blocking=True), 
                                                      return_features=True)
                feat_batch = feat_batch.reshape(size, -1)
            else:
                # Extract features from layer used for TAC measurement
                pred_batch, feat_batch = model.forward_all_features(in_batch.to(device, non_blocking=True))
                feat_batch = feat_batch[-1]

            features = torch.cat([features, feat_batch.cpu()])
            predictions = torch.cat([predictions, pred_batch.cpu().argmax(dim=1)])

    return predictions, features


def experiment_variable_identifier(model_arch, dataset, poison_rate):
    """
    Creates a standardized string identifier for experiment configurations.
    
    This identifier is used for organizing saved models, results, and figures.
    
    Args:
        model_arch (str): Model architecture name (e.g., 'resnet18', 'vgg16')
        dataset (str): Dataset name (e.g., 'cifar10', 'cifar100')
        poison_rate (float or None): Poisoning rate (e.g., 0.05 for 5%)
        
    Returns:
        str: Identifier in format '{arch}_{dataset}_p{rate}' or '{arch}_{dataset}_pNone'
        
    Example:
        >>> experiment_variable_identifier('resnet18', 'cifar10', 0.05)
        'resnet18_cifar10_p0-05'
        >>> experiment_variable_identifier('vgg16', 'cifar100', None)
        'vgg16_cifar100_pNone'
    """
    if poison_rate is None:
        return f"{model_arch}_{dataset}_pNone"
    else:
        # Replace decimal point with dash for filesystem compatibility
        poison_rate_str = "-".join(str(poison_rate).split("."))
        return f"{model_arch}_{dataset}_p{poison_rate_str}"


def list_intersection(a, b):
    """
    Returns the intersection of two lists.
    
    Args:
        a (list): First list
        b (list): Second list
        
    Returns:
        list: Elements present in both lists
        
    Example:
        >>> list_intersection(['a', 'b', 'c'], ['b', 'c', 'd'])
        ['b', 'c']
    """
    return list(set(a) & set(b))


def dict_subset(dictionary, keys):
    """
    Creates a new dictionary with only the specified keys.
    
    Args:
        dictionary (dict): Source dictionary
        keys (iterable): Keys to include in subset
        
    Returns:
        dict: New dictionary containing only specified keys
        
    Example:
        >>> d = {'a': 1, 'b': 2, 'c': 3}
        >>> dict_subset(d, ['a', 'c'])
        {'a': 1, 'c': 3}
    """
    return {k: v for k, v in dictionary.items() if k in keys}


# =============================================================================
# SECTION 3: INPUT-SPACE STEALTHINESS METRICS
# =============================================================================

"""
Input-space metrics measure the perceptual similarity between clean and poisoned images.

All metrics in this section have the following signature:
    Args:
        img1 (torch.Tensor): Batch of images, shape (N, 3, H, W), values in [0, 1]
        img2 (torch.Tensor): Batch of images, same shape as img1
    
    Returns:
        float: Average metric value over the batch

Lower values generally indicate higher stealthiness (more similar images),
except for SSIM, PSNR, and pHash where higher values indicate higher stealthiness.
"""


def _initialize_input_space_models(device=None):
    """
    Initializes neural network models used by input-space metrics.
    
    This function should be called once before using LPIPS or IS metrics.
    
    Args:
        device (torch.device, optional): Device to load models on
        
    Returns:
        tuple: (lpips_model, inception_v3_model)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # LPIPS model (AlexNet-based)
    loss_fn = lpips.LPIPS(net="alex", verbose=False).to(device, non_blocking=True)
    
    # Inception v3 for Inception Score
    inception_v3 = torch.hub.load("pytorch/vision:v0.10.0", "inception_v3", 
                                  pretrained=True).to(device, non_blocking=True)
    inception_v3.eval()
    
    return loss_fn, inception_v3


# Global models for efficiency (initialized on first use)
_LPIPS_MODEL = None
_INCEPTION_V3_MODEL = None


def _get_input_space_models(device=None):
    """Lazy initialization of input-space metric models."""
    global _LPIPS_MODEL, _INCEPTION_V3_MODEL
    if _LPIPS_MODEL is None or _INCEPTION_V3_MODEL is None:
        _LPIPS_MODEL, _INCEPTION_V3_MODEL = _initialize_input_space_models(device)
    return _LPIPS_MODEL, _INCEPTION_V3_MODEL


def lp_norm(p, imgs):
    """
    Computes the Lp norm of images.
    
    Args:
        p (int or float): Order of the norm (1, 2, np.inf, etc.)
        imgs (torch.Tensor or numpy.ndarray): Images to compute norm of
        
    Returns:
        float: Mean Lp norm across all images
    """
    return np.mean([norm(img.flatten(), ord=p) for img in imgs])


def l1_distance(img1, img2):
    """
    L1 (Manhattan) distance between two image batches.
    
    Measures the sum of absolute differences between pixel values.
    
    Range: [0, ∞), where 0 indicates identical images
    """
    return lp_norm(1, img1 - img2)


def l2_distance(img1, img2):
    """
    L2 (Euclidean) distance between two image batches.
    
    Measures the root mean square difference between pixel values.
    
    Range: [0, ∞), where 0 indicates identical images
    """
    return lp_norm(2, img1 - img2)


def linf_distance(img1, img2):
    """
    L∞ (Chebyshev) distance between two image batches.
    
    Measures the maximum absolute difference between any pair of pixels.
    
    Range: [0, 1], where 0 indicates identical images
    """
    return lp_norm(np.inf, img1 - img2)


def MSE(img1, img2):
    """
    Mean Squared Error between two image batches.
    
    Measures the average squared difference between pixel values.
    
    Range: [0, 1], where 0 indicates identical images
    """
    return mean_squared_error(img1.flatten(), img2.flatten())


def PSNR(img1, img2):
    """
    Peak Signal-to-Noise Ratio between two image batches.
    
    Measures the ratio between maximum possible signal power and corrupting noise power.
    Higher values indicate higher similarity.
    
    Range: [0, ∞), where ∞ indicates identical images (though practically capped)
    
    Note:
        Returns infinity if images are identical (handled by numpy's error state)
    """
    with np.errstate(divide="ignore"):  # Ignore divide-by-zero warnings
        return peak_signal_noise_ratio(img1.numpy(), img2.numpy(), data_range=1)


def SSIM(img1, img2):
    """
    Structural Similarity Index between two image batches.
    
    Measures perceived quality by considering luminance, contrast, and structure.
    Higher values indicate higher similarity.
    
    Range: [-1, 1], where 1 indicates identical images
    
    Reference:
        Wang et al., "Image Quality Assessment: From Error Visibility to
        Structural Similarity", IEEE TIP 2004
    """
    SSIM_per_image = lambda x, y: structural_similarity(x, y, data_range=1, channel_axis=0)
    SSIM_values = np.array(list(map(SSIM_per_image, img1.numpy(), img2.numpy())))
    return SSIM_values.mean()


def pHash(img1, img2):
    """
    Perceptual Hash similarity between two image batches.
    
    Computes perceptual hashes and measures their Hamming distance.
    Robust to minor modifications like scaling and color adjustments.
    Higher values indicate higher similarity.
    
    Range: [0, 100], where 100 indicates identical perceptual hashes
    
    Returns:
        float: Percentage of matching hash bits
        
    Note:
        pHash uses an 8x8 hash (64 bits), so similarity is computed as:
        100 * (1 - hamming_distance / 64)
    """
    def pHash_per_image(x, y):
        tensor_to_pil = T.ToPILImage()
        pil1 = tensor_to_pil(x)
        pil2 = tensor_to_pil(y)

        hash1 = phash(pil1)
        hash2 = phash(pil2)
    
        def hamming_distance(hash1, hash2):
            # Convert 8x8 boolean hash to flat array
            bool_array1 = hash1.hash.flatten()
            bool_array2 = hash2.hash.flatten()
            return np.where(bool_array1 != bool_array2)[0].size
        
        return (1 - (hamming_distance(hash1, hash2) / 64.0)) * 100
    
    pHash_values = np.array(list(map(pHash_per_image, img1, img2)))
    return pHash_values.mean()


def LPIPS(img1, img2, device=None):
    """
    Learned Perceptual Image Patch Similarity.
    
    Uses a pretrained AlexNet to measure perceptual similarity in feature space.
    Lower values indicate higher perceptual similarity.
    
    Range: [0, ∞) in theory, typically [0, 1] in practice
    
    Reference:
        Zhang et al., "The Unreasonable Effectiveness of Deep Features as a
        Perceptual Metric", CVPR 2018
        
    Note:
        Requires images to be normalized to [-1, 1] internally
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    loss_fn, _ = _get_input_space_models(device)
    
    with torch.no_grad():
        normalize = T.Normalize([0.5], [0.5])  # [0,1] -> [-1,1]
        img1_norm = normalize(img1.to(device, non_blocking=True))
        img2_norm = normalize(img2.to(device, non_blocking=True))
        return loss_fn.forward(img1_norm, img2_norm).mean().item()


def IS(img1, img2, device=None):
    """
    Inception Score difference (using KL divergence).
    
    Measures the difference in class probability distributions predicted by
    Inception v3. Lower values indicate more similar semantic content.
    
    Range: [0, ∞), where 0 indicates identical predicted distributions
    
    Returns:
        float: Mean KL divergence between predicted distributions
        
    Reference:
        Salimans et al., "Improved Techniques for Training GANs", NeurIPS 2016
        
    Note:
        This is a repurposing of Inception Score for similarity measurement,
        computed as KL(P1 || P2) where P1, P2 are prediction distributions
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    _, inception_v3 = _get_input_space_models(device)
    
    with torch.no_grad():
        # Inception v3 expects 299x299 images, normalized with ImageNet stats
        preprocess = T.Compose([
            T.Resize(299),
            T.CenterCrop(299),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
               
        out1 = inception_v3(preprocess(img1.to(device, non_blocking=True)))
        out2 = inception_v3(preprocess(img2.to(device, non_blocking=True)))

        # Convert logits to probabilities
        preds1 = softmax(out1, dim=1)
        preds2 = softmax(out2, dim=1)
        
        # Compute KL divergence: sum over classes, mean over batch
        return rel_entr(preds1.cpu().numpy(), preds2.cpu().numpy()).sum(axis=1).mean()


def SAM(img1, img2):
    """
    Spectral Angle Mapper between two image batches.
    
    Measures the spectral similarity by computing the angle between pixel vectors.
    Commonly used in remote sensing; here adapted for RGB images.
    Lower values indicate higher similarity.
    
    Range: [0, π/2] radians, where 0 indicates identical spectral signatures
    
    Reference:
        Kruse et al., "The Spectral Image Processing System (SIPS) - Interactive
        Visualization and Analysis of Imaging Spectrometer Data", Remote Sensing
        of Environment 1993
        
    Note:
        Images are clamped to [1e-8, 1] to avoid NaN from zero vectors
    """
    # Avoid zero vectors which cause NaN
    img1_clamped = torch.clamp(img1, 1e-8, 1)
    img2_clamped = torch.clamp(img2, 1e-8, 1)

    return spectral_angle_mapper(img1_clamped, img2_clamped, 
                                 reduction='elementwise_mean').item()


# =============================================================================
# SECTION 4: FEATURE-SPACE STEALTHINESS METRICS
# =============================================================================

"""
Feature-space metrics analyze the latent representations of clean and poisoned samples.

These metrics typically operate on features extracted from intermediate layers of
neural networks and assess the separability of poisoned samples from benign ones.
"""


def create_tsne(trainset, feature_path, skip_misclassified=False, show_plot=False, 
               save_dst=None):
    """
    Creates t-SNE visualization of feature space separability.
    
    This function performs dimensionality reduction on extracted features and
    splits them into benign, poisoned, and cross-trigger clusters for visualization.
    
    Args:
        trainset (BackdoorDataset): Training dataset with poison_lookup and cross_lookup
        feature_path (str): Path to .pt file containing:
            - 'predictions': Model predictions on target-class samples
            - 'features': Extracted features (N, feature_dim)
            - 'indices': Indices of samples in trainset
        skip_misclassified (bool, optional): If True, exclude misclassified samples.
                                            Default: False
        show_plot (bool, optional): If True, display plot. Default: False
        save_dst (str, optional): Directory to save plots. Default: None (no saving)
        
    Returns:
        tuple: (features_benign, features_poisoned, gt_labels_poisoned) where
            - features_benign: 2D array of t-SNE embeddings for benign samples
            - features_poisoned: 2D array of t-SNE embeddings for poisoned samples
            - gt_labels_poisoned: Original class labels of poisoned samples
            
    Saves:
        If save_dst is provided, saves plots with various visualization options:
        - default.png: Basic benign (blue) vs poisoned (red)
        - groupby_gt.png: Poisoned samples colored by original class
        
    Example:
        >>> trainset = BackdoorDataset(...)
        >>> benign_feats, poison_feats, labels = create_tsne(
        ...     trainset,
        ...     'results/features/badnet_p0.05.pt',
        ...     save_dst='results/tsne/badnet'
        ... )
    """
    # Load extracted features and metadata
    feature_dict = torch.load(feature_path, weights_only=False)
    predictions = feature_dict["predictions"]
    features = feature_dict["features"]
    target_label_indices = feature_dict["indices"]
    
    # Optionally filter misclassified samples
    if skip_misclassified:
        correctly_classified = predictions == trainset.poisoned_labels[target_label_indices]
        features = features[correctly_classified]
        target_label_indices = target_label_indices[correctly_classified]

    # Perform dimensionality reduction with t-SNE
    print("Running t-SNE dimensionality reduction...")
    features_embedded = TSNE().fit_transform(features)

    # Split features into benign, poisoned, and cross-trigger
    def split_features(features, poison_lookup, cross_lookup, indices):
        is_poisoned = poison_lookup[indices]
        is_cross = cross_lookup[indices]
        is_benign = np.logical_not(np.logical_or(is_poisoned, is_cross))

        benign = features[is_benign]
        poisoned = features[is_poisoned]
        cross = features[is_cross]
        
        return benign, poisoned, cross, is_poisoned

    features_benign, features_poisoned, features_cross, is_poisoned = split_features(
        features_embedded, trainset.poison_lookup, trainset.cross_lookup, target_label_indices
    )
    gt_labels = trainset.original_labels[target_label_indices][is_poisoned]

    # Visualization function
    def plot(features_benign, features_poisoned, features_cross, gt_labels, 
            highlight_cross=False, groupby_gt=False, show_plot=False, save_dst=None):
        """Helper function to create t-SNE scatter plots."""
        def plot_benign(features):
            plt.scatter(features[:, 0], features[:, 1], label="Benign", 
                       marker='o', s=5, color="blue", alpha=1.0)

        # Plot cross samples separately or merge with benign
        if highlight_cross and len(features_cross) > 0:
            plot_benign(features_benign)
            plt.scatter(features_cross[:, 0], features_cross[:, 1], label="Cross", 
                       marker='v', s=8, color='green', alpha=0.7)
        else:
            plot_benign(np.concatenate([features_benign, features_cross], axis=0))

        # Group poisoned samples by original class for detailed analysis
        if groupby_gt:
            class_colors = ['#B82B22', '#CE2658', '#E11F8D', '#F114C5', '#FF00FF',
                           '#CA5E23', '#DA8723', '#E8AF1F', '#F4D717', '#FFFF00']

            for i, c in enumerate(trainset.classes):
                c_indices = gt_labels == i
                if not np.any(c_indices):
                    continue
                c_features = features_poisoned[c_indices]
                plt.scatter(c_features[:, 0], c_features[:, 1], 
                           label=f"Poisoned ({c})", marker='^', s=8, 
                           color=class_colors[i], alpha=0.7)
        else:
            plt.scatter(features_poisoned[:, 0], features_poisoned[:, 1], 
                       label="Poisoned", marker='^', s=8, color="red", alpha=0.7)
            
        plt.axis("off")
        plt.tight_layout()

        if save_dst:
            os.makedirs(save_dst, exist_ok=True)
            opts = np.array([highlight_cross, groupby_gt])
            opts_str = np.array(["highlight_cross", "groupby_gt"])[opts]
            filename = f"{'_'.join(opts_str)}" if len(opts_str) > 0 else "default"
            plt.savefig(os.path.join(save_dst, filename + ".png"), transparent=True)

        if show_plot:
            plt.legend()
            plt.show()
            plt.clf()
        else:
            plt.close()
    
    # Generate plots with various options
    if show_plot or save_dst:
        highlight_cross = False
        for groupby_gt in [False, True]:
            # Skip class-grouped plots if too many classes
            if groupby_gt and len(trainset.classes) != 10:
                continue
            plot(features_benign, features_poisoned, features_cross, gt_labels, 
                highlight_cross, groupby_gt, show_plot, save_dst)

    # Return concatenated benign + cross features (both considered benign)
    return np.concatenate([features_benign, features_cross]), features_poisoned, gt_labels


def clustering_score(features_benign, features_poisoned, silhouette=True):
    """
    Computes clustering quality score for benign vs poisoned features.
    
    Args:
        features_benign (numpy.ndarray): Benign sample features (N1, feature_dim)
        features_poisoned (numpy.ndarray): Poisoned sample features (N2, feature_dim)
        silhouette (bool, optional): If True, compute Silhouette Score;
                                     otherwise compute Davies-Bouldin Index. Default: True
        
    Returns:
        float: Clustering score
            - Silhouette Score: [-1, 1], higher is better (better separation)
            - Davies-Bouldin Index: [0, ∞), lower is better (better separation)
            
    Note:
        Both metrics assess how well the two clusters (benign vs poisoned) are separated:
        - Higher Silhouette Score → poisoned samples form distinct cluster (less stealthy)
        - Lower DBI → poisoned samples form distinct cluster (less stealthy)
    """
    n_benign = len(features_benign)
    n_poisoned = len(features_poisoned)
    
    # Create cluster labels: 0 for benign, 1 for poisoned
    cluster_labels = np.concatenate([np.zeros(n_benign), np.ones(n_poisoned)])
    features = np.concatenate([features_benign, features_poisoned])
    
    if silhouette:
        return silhouette_score(features, cluster_labels)
    else:
        return davies_bouldin_score(features, cluster_labels)


def SS(features_benign, features_poisoned):
    """
    Silhouette Score for feature-space separability.
    
    Convenient wrapper around clustering_score with silhouette=True.
    
    Range: [-1, 1], where higher values indicate more separable clusters
    (less stealthy backdoor)
    
    Args:
        features_benign (numpy.ndarray): Benign sample features
        features_poisoned (numpy.ndarray): Poisoned sample features
        
    Returns:
        float: Silhouette Score
    """
    return clustering_score(features_benign, features_poisoned, silhouette=True)


def CDBI(features_benign, features_poisoned, gt_labels_poisoned, n_classes):
    """
    Class-specific Davies-Bouldin Index (CDBI).
    
    A novel metric introduced in the paper. Computes the average DBI between
    benign features and each class-specific poisoned subcluster. This provides
    a more fine-grained analysis than overall clustering metrics.
    
    Args:
        features_benign (numpy.ndarray): Benign sample features (N1, feature_dim)
        features_poisoned (numpy.ndarray): Poisoned sample features (N2, feature_dim)
        gt_labels_poisoned (numpy.ndarray): Original class labels of poisoned samples (N2,)
        n_classes (int): Total number of classes in the dataset
        
    Returns:
        tuple: (mean_cdbi, cdbi_per_class) where
            - mean_cdbi: Average DBI across all represented classes
            - cdbi_per_class: List of DBI values for each class with poisoned samples
            
    Note:
        - Lower CDBI indicates more stealthy backdoor (less separable subclusters)
        - Classes without poisoned samples are skipped (common for target class
          in clean-label attacks and DFST)
          
    Example:
        >>> mean_cdbi, class_dbis = CDBI(benign_feats, poison_feats, labels, n_classes=10)
        >>> print(f"Average CDBI: {mean_cdbi:.3f}")
        >>> print(f"Per-class DBIs: {class_dbis}")
    """
    dbis = []

    for c in range(n_classes):
        c_indices = gt_labels_poisoned == c

        # Skip classes without poisoned samples
        if not np.any(c_indices):
            continue

        # Compute DBI for this class's poisoned samples vs all benign samples
        dbi = clustering_score(features_benign, features_poisoned[c_indices], 
                              silhouette=False)
        dbis.append(dbi)

    return np.mean(dbis), dbis


def DSWD_eq_7(features_clean, features_bd, param_matrix):
    """
    Implementation of Discriminant Sliced-Wasserstein Distance (Equation 7).
    
    Computes DSWD by projecting features using the classifier's weight matrix
    and measuring Wasserstein distance along each projection direction.
    
    Args:
        features_clean (torch.Tensor): Clean sample features (N, d)
        features_bd (torch.Tensor): Backdoored sample features (N, d)
        param_matrix (torch.Tensor): Classifier weight matrix (C, d) where
                                     C is number of classes, d is feature dimension
        
    Returns:
        float: DSWD value
        
    Formula:
        DSWD = sqrt( (1/|C|) * Σ_c W_2(w_c^T F_c, w_c^T F_b)^2 )
        
        where:
        - |C| is the number of classes
        - w_c is the normalized weight vector for class c
        - F_c, F_b are clean and backdoored feature matrices
        - W_2 is the 2-Wasserstein distance
        
    Reference:
        Tang et al., "Backdoor Attack with Imperceptible Input and Latent
        Modification", arXiv 2022
        
    Note:
        Lower DSWD indicates more stealthy backdoor (smaller feature distribution shift)
    """
    # Normalize weight vectors (rows of param_matrix)
    param_matrix_norm = normalize(param_matrix, axis=1)
    
    # Project features onto weight directions
    features_clean = features_clean.T
    features_bd = features_bd.T
    projected_features_clean = np.matmul(param_matrix_norm, features_clean.numpy())
    projected_features_bd = np.matmul(param_matrix_norm, features_bd.numpy())

    # Compute sum of squared 2-Wasserstein distances
    classes = range(len(param_matrix))
    sum_wasserstein = np.sum([
        wasserstein_1d(projected_features_clean[c], projected_features_bd[c], p=2)
        for c in classes
    ])
    
    # Compute DSWD: sqrt of average
    dswd = (sum_wasserstein / len(classes)) ** 0.5

    return dswd


def DSWD(model, feature_path, skip_misclassified=False, model_arch="resnet18"):
    """
    Discriminant Sliced-Wasserstein Distance for feature-space analysis.
    
    This is a convenience wrapper that loads features from a file and
    extracts the classifier weights automatically.
    
    Args:
        model (torch.nn.Module): Trained model
        feature_path (str): Path to .pt file containing:
            - 'features_clean': Clean sample features
            - 'features_bd': Backdoored sample features
            - 'predictions_bd': Predictions on backdoored samples
        skip_misclassified (bool, optional): If True, exclude misclassified
                                            backdoored samples. Default: False
        model_arch (str, optional): Model architecture name for determining
                                   final layer name. Default: 'resnet18'
        
    Returns:
        float: DSWD value
        
    Example:
        >>> dswd_value = DSWD(
        ...     model=backdoored_model,
        ...     feature_path='results/features/badnet_p0.05.pt',
        ...     model_arch='resnet18'
        ... )
        >>> print(f"DSWD: {dswd_value:.4f}")
    """
    # Determine final layer name based on architecture
    if model_arch == "resnet18":
        final_layer_name = "linear"
    elif model_arch == "vit_small":
        final_layer_name = "head"
    else:  # VGG and others
        final_layer_name = "classifier"
    
    weights = model.state_dict()[f"{final_layer_name}.weight"].cpu()

    # Load features
    feature_dict = torch.load(feature_path, weights_only=False)
    features_clean = feature_dict["features_clean"]
    features_bd = feature_dict["features_bd"]
    predictions = feature_dict["predictions_bd"]
    
    # Optionally filter misclassified samples
    if skip_misclassified:
        target_class = torch.mode(predictions).values.item()  # Infer target class
        correctly_classified = predictions == target_class
        features_clean = features_clean[correctly_classified]
        features_bd = features_bd[correctly_classified]

    return DSWD_eq_7(features_clean, features_bd, weights)


# =============================================================================
# SECTION 5: PARAMETER-SPACE STEALTHINESS METRICS
# =============================================================================

"""
Parameter-space metrics analyze the model's learned parameters to detect
backdoor-related artifacts.

These metrics examine properties like channel sensitivity (UCLC) and
activation changes (TAC) to identify potentially backdoored neurons.
"""


def UCLC(model, normalize=True):
    """
    Upper bound of Channel Lipschitzness Constant.
    
    Computes the sensitivity of each convolutional channel by analyzing
    the singular values of combined convolution-batchnorm weights.
    Backdoored channels typically have anomalously high UCLC values.
    
    Args:
        model (torch.nn.Module): Neural network model (or submodule)
        normalize (bool, optional): If True, normalize UCLC values per layer
                                   (z-score normalization). Default: True
        
    Returns:
        torch.Tensor: 1D tensor of UCLC values for each channel across all layers
        
    Note:
        - Higher UCLC indicates higher channel sensitivity (potential backdoor indicator)
        - Typically, the maximum UCLC value is used as the metric
        - Requires model to have Conv2d layers followed by BatchNorm2d layers
        
    Reference:
        Wu et al., "Adversarial Neuron Pruning Purifies Backdoored Deep Models", NeurIPS 2021
        Official implementation: https://github.com/rkteddy/channel-Lipschitzness-based-pruning
        
    Example:
        >>> model = ResNet18(num_classes=10)
        >>> uclc = UCLC(model)
        >>> max_uclc = uclc.max().item()
        >>> print(f"Maximum UCLC: {max_uclc:.2f}")
    """
    uclc = torch.Tensor([])
    conv = None
    
    for name, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            # Extract BatchNorm parameters
            std = m.running_var.sqrt()
            weight = m.weight

            channel_lips = []
            for idx in range(weight.shape[0]):
                # Combine convolution and BatchNorm weights
                # W_combined = W_conv * (γ / σ) where γ is BN weight, σ is std
                w = conv.weight[idx].reshape(conv.weight.shape[1], -1) * \
                    (weight[idx] / std[idx]).abs()
                
                # Channel Lipschitzness = largest singular value
                channel_lips.append(torch.svd(w.cpu())[1].max())
            
            channel_lips = torch.Tensor(channel_lips)

            # Normalize within layer to highlight anomalies
            if normalize:
                channel_lips = (channel_lips - channel_lips.mean()) / channel_lips.std()

            uclc = torch.cat((uclc, channel_lips))
        
        # Store convolution layer (should precede BatchNorm)
        elif isinstance(m, nn.Conv2d):
            conv = m

    return uclc


def TAC(tac_path, bd_or_clean="bd", skip_misclassified=False, target_class=0):
    """
    Trigger-Activated Change (TAC).
    
    Measures the per-neuron activation difference between clean and backdoored
    images. Backdoored models typically have neurons with anomalously high TAC.
    
    Args:
        tac_path (str): Path to .pt file containing:
            - 'bd': Dict with 'tac' (per-input TAC) and 'predictions_bd'
            - 'clean': Dict with 'tac' and 'predictions_bd' (for benign model baseline)
        bd_or_clean (str, optional): Which model's TAC to return ('bd' or 'clean').
                                     Default: 'bd'
        skip_misclassified (bool, optional): If True, exclude misclassified samples.
                                            Default: False
        target_class (int, optional): Expected target class for poisoned samples.
                                     Default: 0
        
    Returns:
        torch.Tensor: 1D tensor of average TAC per neuron (shape: [num_neurons])
        
    Note:
        - Higher TAC indicates neurons strongly activated by the trigger
        - TAC is computed from a specific intermediate layer (not penultimate)
        - Comparing backdoored vs clean model TAC helps identify trigger-specific neurons
        
    Reference:
        Tang et al., "Towards Backdoor Stealthiness in Model Parameter Space", arXiv 2025
        
    Example:
        >>> tac_bd = TAC('results/tac/badnet_p0.05.pt', bd_or_clean='bd')
        >>> tac_clean = TAC('results/tac/badnet_p0.05.pt', bd_or_clean='clean')
        >>> tac_ratio = tac_bd / tac_clean
        >>> print(f"Top 5 suspicious neurons: {tac_ratio.argsort()[-5:]}")
    """
    # Load TAC data
    tac_dict = torch.load(tac_path, weights_only=False)
    tac_per_input = tac_dict[bd_or_clean]["tac"]
    predictions = tac_dict[bd_or_clean]["predictions_bd"]

    # Optionally filter misclassified samples
    if skip_misclassified:
        correctly_classified = predictions == target_class
        tac_per_input = tac_per_input[correctly_classified]

    # Return average TAC per neuron across all inputs
    return tac_per_input.mean(dim=0)


def TAC_comparison(record_dict, model_arch, dataset, attacks, poison_rates,
                  result_dir="results", target_class=0):
    """
    Creates comparison plots of TAC values for multiple attacks.
    
    Visualizes TAC distributions for both backdoored and benign models,
    helping to identify attacks with more stealthy parameter-space footprints.
    
    Args:
        record_dict (dict): Nested dict of attack records:
            {model_arch: {dataset: {attack: {poison_rate: {'model': ...}}}}}
        model_arch (str): Model architecture ('resnet18', 'vgg16', etc.)
        dataset (str): Dataset name ('cifar10', 'cifar100', etc.)
        attacks (list): List of attack names to compare
        poison_rates (list): List of poison rate functions
        result_dir (str, optional): Directory containing TAC files. Default: 'results'
        target_class (int, optional): Target class. Default: 0
        
    Saves:
        Comparison plot to: {result_dir}/tac_activations/{arch}_{dataset}/TAC_comparison.png
        
    Example:
        >>> TAC_comparison(
        ...     record_dict=records,
        ...     model_arch='resnet18',
        ...     dataset='cifar10',
        ...     attacks=['badnet', 'blend', 'wanet'],
        ...     poison_rates=[lambda x: 0.05],
        ...     result_dir='./results'
        ... )
    """
    def plot_TAC(tac_save_path, ax, atk, pr):
        """Helper to plot TAC for one attack."""
        # Pretty attack name
        ATK_PPRINT_DICT = {
            "badnet": "BadNets", "blended": "Blend", "wanet": "WaNet",
            "bpp": "BppAttack", "adaptive_patch": "Adap-Patch",
            "adaptive_blend": "Adap-Blend", "dfst": "DFST", "dfba": "DFBA",
            "narcissus": "Narcissus", "grond": "Grond"
        }
        bd_label = f"{ATK_PPRINT_DICT.get(atk, atk)}"
        if pr:
            bd_label += f" {pr*100:g}% PR"

        # Plot TAC for both backdoored and benign models
        for desc in ["bd", "clean"]:
            tac = TAC(tac_save_path, desc, target_class=target_class)
            label = "Benign" if desc == "clean" else bd_label
            color = "blue" if desc == "clean" else "red"

            ax.scatter(range(len(tac)), np.sort(tac), label=label, 
                      s=4, color=color, alpha=0.7)
            ax.legend(loc="upper left")
            ax.set_yscale('log')  # Log scale for wide range

    # Determine grid layout
    row_multiplier = 2 if model_arch == "resnet18" else 1
    n_rows = row_multiplier * len(poison_rates)
    n_plots = len(attacks) * len(poison_rates)
    n_cols = n_plots // n_rows
    
    fig, ax = plt.subplots(n_rows, n_cols, sharex=True, sharey=True, squeeze=False)
    fig.set_figheight(2.5 * n_rows)
    fig.set_figwidth(3.5 * n_cols)
    exp_id = f"{model_arch}_{dataset}"

    # Plot each attack
    for i, atk in enumerate(attacks):
        col_idx = i % n_cols
        row_idx = 0 if i < n_cols else len(poison_rates)
        atk_dict = record_dict[model_arch][dataset][atk]

        if atk == "dfba":
            tac_save_path = os.path.join(result_dir, "tac_activations", exp_id, f"{atk}.pt")
            plot_TAC(tac_save_path, ax[row_idx][col_idx], atk, None)
            if len(poison_rates) > 1:
                fig.delaxes(ax[row_idx+1][col_idx])  # Remove unused subplot
        else:
            for j, pr in enumerate(atk_dict.keys()):
                tac_save_path = os.path.join(result_dir, "tac_activations", 
                                            exp_id, f"{atk}_p{pr}.pt")
                plot_TAC(tac_save_path, ax[row_idx + j][col_idx], atk, pr)

    fig.tight_layout()
    save_path = os.path.join(result_dir, "tac_activations", exp_id, "TAC_comparison.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300)
    print(f"TAC comparison saved to {save_path}")
    plt.close(fig)


def TUP(tac_path, model_bd, model_arch="resnet18"):
    """
    TAC-UCLC Product (TUP).
    
    A novel metric introduced in the paper. Combines TAC and UCLC to identify
    backdoored neurons that are both trigger-activated (high TAC) and sensitive
    (high UCLC). The product is weighted by the ratio of backdoored to benign TAC.
    
    Args:
        tac_path (str): Path to .pt file containing TAC values for both
                       backdoored and clean models
        model_bd (torch.nn.Module): Backdoored model
        model_arch (str, optional): Model architecture. Default: 'resnet18'
        
    Returns:
        float: Weighted average TUP value
        
    Formula:
        TUP = mean( UCLC_bd[i] * (TAC_bd[i] / TAC_clean[i]) )
        
        where i iterates over neurons, sorted by backdoored TAC
        
    Note:
        - Higher TUP indicates more detectable backdoor
        - TUP combines feature-space (TAC) and parameter-space (UCLC) signals
        - Focuses on the same layer for both TAC and UCLC
        
    Example:
        >>> tup_value = TUP(
        ...     tac_path='results/tac/badnet_p0.05.pt',
        ...     model_bd=backdoored_model,
        ...     model_arch='resnet18'
        ... )
        >>> print(f"TUP: {tup_value:.2f}")
    """
    # Extract average TAC for backdoored and clean models
    tac_bd = TAC(tac_path, bd_or_clean="bd")
    tac_clean = TAC(tac_path, bd_or_clean="clean")

    # Sort TAC values (preserve backdoored model's ordering)
    bd_indices = np.argsort(tac_bd)
    tac_bd_sorted = tac_bd[bd_indices]
    tac_clean_sorted = np.sort(tac_clean)

    # Compute TAC ratio (backdoor vs benign)
    tac_ratio = tac_bd_sorted / tac_clean_sorted

    # Extract the relevant layer for TAC computation
    # (same layer used for UCLC to ensure consistency)
    if model_arch == "resnet18":
        model_bd = model_bd.layer4[1]
        model_bd = nn.Sequential(model_bd.conv2, model_bd.bn2)
    else:  # VGG and others
        model_bd = model_bd.features[-5:-3]

    # Compute UCLC and reorder to match TAC ordering
    uclc_bd = UCLC(model_bd, normalize=False)[bd_indices]

    # Return weighted average
    return np.average(uclc_bd * tac_ratio)


# =============================================================================
# SECTION 6: DATA AND MODEL LOADING FUNCTIONS
# =============================================================================


def get_dataset(dataset_name, train=True, transforms=None, data_dir="data"):
    """
    Loads a dataset by name.
    
    Args:
        dataset_name (str): One of 'cifar10', 'cifar100', 'imagenette'
        train (bool, optional): If True, load training set; else test set. Default: True
        transforms (callable, optional): Transforms to apply to images
        data_dir (str, optional): Root directory for datasets. Default: 'data'
        
    Returns:
        Dataset: PyTorch dataset object
        
    Example:
        >>> train_data = get_dataset('cifar10', train=True, transforms=my_transform)
        >>> test_data = get_dataset('cifar100', train=False)
    """
    data_root = os.path.join(data_dir, dataset_name)

    if dataset_name == "cifar10":
        return torchvision.datasets.CIFAR10(
            root=data_root, train=train, download=True, transform=transforms
        )
    elif dataset_name == "cifar100":
        return torchvision.datasets.CIFAR100(
            root=data_root, train=train, download=True, transform=transforms
        )
    elif dataset_name == "imagenette":
        return Imagenette(root=data_root, train=train, transform=transforms)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def load_model_state(arch, dataset, state_dict, device=None):
    """
    Loads a model with specified architecture and restores its state.
    
    Args:
        arch (str): Model architecture ('resnet18', 'vgg16', etc.)
        dataset (str): Dataset name (determines number of classes)
        state_dict (dict): Model state dictionary
        device (torch.device, optional): Device to load model on
        
    Returns:
        torch.nn.Module: Loaded model in eval mode
        
    Supported Architectures:
        - 'resnet18': ResNet-18 (from BackdoorBench)
        - 'vgg16': VGG-16 (from BackdoorBench)
        
    Example:
        >>> state_dict = torch.load('model.pth')
        >>> model = load_model_state('resnet18', 'cifar10', state_dict)
        >>> predictions = model(images)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Determine number of classes
    n_classes_dict = {
        "cifar10": 10,
        "cifar100": 100,
        "imagenette": 10
    }
    num_classes = n_classes_dict.get(dataset, 10)

    # Import model architectures (adjust import path as needed)
    try:
        sys.path.append(os.path.abspath("./backdoorbench"))
        from backdoorbench.models.resnet import ResNet18
        from backdoorbench.models.vgg import VGG16
    except ImportError:
        raise ImportError("BackdoorBench models not found. Ensure ./backdoorbench is available.")

    # Create model
    if arch == "resnet18":
        model = ResNet18(num_classes=num_classes)
    elif arch == "vgg16":
        model = VGG16(num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported architecture: {arch}")
    
    # Load state and set to eval mode
    model.load_state_dict(state_dict)
    model.eval()
    return model.to(device, non_blocking=True)


def load_clean_record(dataset, arch, record_dir="record", data_dir="data", 
                     transform_dict=None, target_class=0):
    """
    Loads clean (benign) dataset and model.
    
    Args:
        dataset (str): Dataset name ('cifar10', 'cifar100', etc.)
        arch (str): Model architecture ('resnet18', 'vgg16')
        record_dir (str, optional): Directory containing model checkpoints. Default: 'record'
        data_dir (str, optional): Directory containing datasets. Default: 'data'
        transform_dict (dict, optional): Dict mapping '{dataset}_{split}' to transforms
        target_class (int, optional): Target class to filter from test sets. Default: 0
        
    Returns:
        dict: Dictionary containing:
            - 'train': Training dataset (with transforms)
            - 'test': Test dataset (target class filtered, with transforms)
            - 'train_transformed': Training dataset (augmented transforms)
            - 'test_transformed': Test dataset (target class filtered, augmented)
            - 'model': Loaded clean model
            
    Example:
        >>> clean_record = load_clean_record(
        ...     dataset='cifar10',
        ...     arch='resnet18',
        ...     transform_dict=my_transforms,
        ...     target_class=0
        ... )
        >>> clean_model = clean_record['model']
        >>> clean_testset = clean_record['test']
    """
    record = {}

    # Load datasets with appropriate transforms
    for key in ["train", "test", "train_transformed", "test_transformed"]:
        is_train = "train" in key
        transform_key = f"{dataset}_{key}"
        transforms = transform_dict.get(transform_key) if transform_dict else None
        
        dataset_obj = get_dataset(dataset, train=is_train, transforms=transforms, 
                                 data_dir=data_dir)

        # Filter target class from test sets
        if key in ["test", "test_transformed"]:
            dataset_obj = filter_target_class(dataset_obj, target_class)

        record[key] = dataset_obj

    # Load clean model
    exp_id = experiment_variable_identifier(arch, dataset, None)
    clean_path = os.path.join(record_dir, f"prototype_{exp_id}", "clean_model.pth")
    state_dict = torch.load(clean_path)
    record["model"] = load_model_state(arch, dataset, state_dict)

    return record


def load_backdoor_record(dataset, arch, atk, poison_rate, clean_record, 
                        record_dir="record"):
    """
    Loads backdoored dataset and model for a specific attack.
    
    Dispatches to attack-specific loading functions based on attack name.
    
    Args:
        dataset (str): Dataset name
        arch (str): Model architecture
        atk (str): Attack name (e.g., 'badnet', 'adaptive_patch', 'dfst')
        poison_rate (float): Poisoning rate (e.g., 0.05 for 5%)
        clean_record (dict): Clean record from load_clean_record()
        record_dir (str, optional): Directory containing attack records. Default: 'record'
        
    Returns:
        dict: Dictionary containing backdoored datasets and model.
              Structure varies by attack, but typically includes:
              - 'train': Backdoored training dataset
              - 'test': Backdoored test dataset
              - 'model': Backdoored model
              
    Supported Attacks:
        - BackdoorBench: badnet, blended, wanet, bpp, narcissus
        - Adaptive: adaptive_patch, adaptive_blend
        - DFST: dfst
        - DFBA: dfba
        - Grond: grond
        
    Example:
        >>> clean_rec = load_clean_record('cifar10', 'resnet18')
        >>> bd_record = load_backdoor_record(
        ...     dataset='cifar10',
        ...     arch='resnet18',
        ...     atk='badnet',
        ...     poison_rate=0.05,
        ...     clean_record=clean_rec
        ... )
        >>> bd_model = bd_record['model']
        >>> bd_testset = bd_record['test']
    """
    exp_id = experiment_variable_identifier(arch, dataset, poison_rate)
    atk_path = os.path.join(record_dir, f"{atk}_{exp_id}")

    # Dispatch to attack-specific loader
    if atk in ["badnet", "blended", "wanet", "bpp", "narcissus"]:
        return load_backdoorbench(atk, atk_path, dataset, arch)
    elif atk in ["adaptive_patch", "adaptive_blend"]:
        return load_adap(atk_path, dataset, arch, clean_record)
    elif atk == "dfst":
        return load_dfst(atk_path, dataset, arch, clean_record)
    elif atk == "grond":
        return load_grond(atk_path, dataset, arch)
    elif atk == "dfba":
        return load_dfba(atk_path, dataset, arch, clean_record)
    else:
        raise ValueError(f"Unsupported attack: {atk}")


def load_backdoorbench(atk, atk_path, dataset, arch, transform_dict=None, target_class=0):
    """
    Loads BackdoorBench attack results.
    
    Args:
        atk (str): Attack name
        atk_path (str): Path to directory containing attack_result.pt
        dataset (str): Dataset name
        arch (str): Model architecture
        transform_dict (dict, optional): Transform dictionary
        target_class (int, optional): Target class. Default: 0
        
    Returns:
        dict: Record with 'train', 'test', 'train_transformed', 'test_transformed', 'model'
    """
    sys.path.append(os.path.abspath("./backdoorbench"))
    from backdoorbench.utils.save_load_attack import load_attack_result

    record = {}
    atk_result = load_attack_result(os.path.join(atk_path, "attack_result.pt"))

    # Load backdoored datasets
    for key in ["train", "test", "train_transformed", "test_transformed"]:
        split = key.split('_')[0]
        bd_dataset = copy.deepcopy(atk_result[f"bd_{split}"])
        
        # Apply transforms if provided
        transform_key = f"{dataset}_{key}"
        transforms = transform_dict.get(transform_key) if transform_dict else None
        
        record[key] = BackdoorBenchDataset(bd_dataset, target_class, 
                                          replace_transform=transforms)

    # Load model
    record["model"] = load_model_state(arch, dataset, atk_result["model"])

    return record


def load_adap(atk_path, dataset, arch, clean_record):
    """
    Loads Adaptive Patch/Blend attack results.
    
    Args:
        atk_path (str): Path to attack record directory
        dataset (str): Dataset name
        arch (str): Model architecture
        clean_record (dict): Clean record with datasets
        
    Returns:
        dict: Record with 'train', 'test', 'train_transformed', 'test_transformed', 'model'
    """
    record = {}

    # Load backdoored data
    for key in ["train", "test", "train_transformed", "test_transformed"]:
        split = key.split('_')[0]
        record[key] = AdapDataset(atk_path, target_class=0, split=split, 
                                 clean_dataset=clean_record[key])
    
    # Load model
    state_dict = torch.load(os.path.join(atk_path, "model.pt"))
    record["model"] = load_model_state(arch, dataset, state_dict)

    return record


def load_dfst(atk_path, dataset, arch, clean_record):
    """
    Loads DFST attack results.
    
    Args:
        atk_path (str): Path to attack record directory
        dataset (str): Dataset name
        arch (str): Model architecture
        clean_record (dict): Clean record with datasets
        
    Returns:
        dict: Record with 'test', 'train_transformed', 'test_transformed', 'model'
              (Note: No untransformed train set for DFST)
    """
    record = {}

    # Load backdoored data (DFST doesn't provide untransformed train)
    for key in ["test", "train_transformed", "test_transformed"]:
        split = key.split('_')[0]
        record[key] = DFSTDataset(atk_path, target_class=0, split=split, 
                                 clean_dataset=clean_record[key])

    # Load model
    state_dict = torch.load(os.path.join(atk_path, "model.pt"), weights_only=False)
    record["model"] = load_model_state(arch, dataset, state_dict)

    return record


def load_dfba(atk_path, dataset, arch, clean_record):
    """
    Loads DFBA attack results.
    
    Args:
        atk_path (str): Path to attack record directory
        dataset (str): Dataset name
        arch (str): Model architecture
        clean_record (dict): Clean record with test datasets
        
    Returns:
        dict: Record with 'test', 'test_transformed', 'model'
              (Note: No train set for data-free DFBA)
    """
    record = {}

    # Load trigger (mask and perturbation)
    mask = torch.load(os.path.join(atk_path, "mask.pth"), weights_only=False)
    delta = torch.load(os.path.join(atk_path, "delta.pth"), weights_only=False)
    
    # Apply trigger to test sets
    for key in ["test", "test_transformed"]:
        record[key] = DFBADataset(clean_record[key], target_class=0, 
                                 delta=delta, mask=mask)
    
    # Load model
    state_dict = torch.load(os.path.join(atk_path, "model.pth"))
    record["model"] = load_model_state(arch, dataset, state_dict)

    return record


def load_grond(atk_path, dataset, arch, transform_dict=None):
    """
    Loads Grond attack results.
    
    Args:
        atk_path (str): Path to attack record directory
        dataset (str): Dataset name
        arch (str): Model architecture
        transform_dict (dict, optional): Transform dictionary
        
    Returns:
        dict: Record with 'test', 'train_transformed', 'test_transformed', 'model'
    """
    record = {}

    # Load backdoored data
    for key in ["test", "train_transformed", "test_transformed"]:
        is_train = "train" in key
        transform_key = f"{dataset}_{key}"
        transforms = transform_dict.get(transform_key) if transform_dict else None
        
        record[key] = GrondDataset(dataset, transforms, target_class=0, 
                                   record_path=atk_path, train=is_train)

    # Load model
    checkpoint = torch.load(os.path.join(atk_path, "checkpoint.pth"))
    state_dict = checkpoint["model"]
    record["model"] = load_model_state(arch, dataset, state_dict)

    return record


# =============================================================================
# SECTION 7: FEATURE EXTRACTION AND SAVING FUNCTIONS
# =============================================================================


def get_sample_size(eval_type, device=None):
    """
    Determines appropriate sample size based on evaluation type and device.
    
    Args:
        eval_type (str): Type of evaluation ('input' or 'model')
        device (torch.device, optional): Computation device
        
    Returns:
        int or None: Sample size, or None for full dataset
        
    Note:
        On GPU, processes full dataset. On CPU, uses smaller samples for efficiency.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Use full dataset on GPU
    if device.type == "cuda":
        return None

    # Use samples on CPU
    if eval_type == "input":
        return 10
    elif eval_type == "model":
        return 100
    else:
        raise ValueError(f"Unknown eval_type: {eval_type}")


def save_train_feature_space(atk_id, model, trainset, path, target_class=0, 
                             sample_size=None, batch_size=100):
    """
    Extracts and saves training set features for target-class samples.
    
    Args:
        atk_id (str): Attack identifier (e.g., 'badnet_p0.05.pt')
        model (torch.nn.Module): Model to extract features from
        trainset (BackdoorDataset): Training dataset
        path (str): Directory to save features
        target_class (int, optional): Target class. Default: 0
        sample_size (int, optional): Number of samples to process. Default: None (all)
        batch_size (int, optional): Batch size. Default: 100
        
    Saves:
        File '{path}/{atk_id}' containing:
        - 'features': Feature vectors (N, feature_dim)
        - 'predictions': Model predictions (N,)
        - 'indices': Sample indices in trainset (N,)
    """
    # Select samples originally of target class or poisoned
    original_target_label = trainset.original_labels == target_class
    target_label_indices = np.argwhere(np.logical_or(original_target_label, 
                                                      trainset.poison_lookup)).squeeze()
    
    if sample_size:
        target_label_indices = np.random.choice(target_label_indices, 
                                               size=sample_size, replace=False)

    subset = torch.utils.data.Subset(trainset, target_label_indices)
    preds, features = extract_preds_and_features(model, subset, batch_size=batch_size)

    train_features = {
        "features": features,
        "predictions": preds,
        "indices": target_label_indices
    }
    
    os.makedirs(path, exist_ok=True)
    torch.save(train_features, os.path.join(path, atk_id))
    print(f"Saved train features to {os.path.join(path, atk_id)}")


def save_test_feature_space(atk_id, model, testset_clean, testset_bd, path, 
                            sample_size=None, batch_size=100):
    """
    Extracts and saves test set features for both clean and backdoored images.
    
    Args:
        atk_id (str): Attack identifier
        model (torch.nn.Module): Model to extract features from
        testset_clean (Dataset): Clean test dataset
        testset_bd (Dataset): Backdoored test dataset
        path (str): Directory to save features
        sample_size (int, optional): Number of samples. Default: None (all)
        batch_size (int, optional): Batch size. Default: 100
        
    Saves:
        File '{path}/{atk_id}' containing:
        - 'features_clean': Features of clean images
        - 'features_bd': Features of backdoored images
        - 'predictions_clean': Predictions on clean images
        - 'predictions_bd': Predictions on backdoored images
        - 'indices': Sample indices
    """
    sample_indices = np.array(range(len(testset_clean)))
    
    if sample_size:
        sample_indices = np.random.choice(sample_indices, size=sample_size, replace=False)
        testset_clean = torch.utils.data.Subset(testset_clean, sample_indices)
        testset_bd = torch.utils.data.Subset(testset_bd, sample_indices)

    preds_clean, features_clean = extract_preds_and_features(model, testset_clean, 
                                                             batch_size=batch_size)
    preds_bd, features_bd = extract_preds_and_features(model, testset_bd, 
                                                       batch_size=batch_size)

    test_features = {
        "features_clean": features_clean,
        "features_bd": features_bd,
        "predictions_clean": preds_clean,
        "predictions_bd": preds_bd,
        "indices": sample_indices
    }
    
    os.makedirs(path, exist_ok=True)
    torch.save(test_features, os.path.join(path, atk_id))
    print(f"Saved test features to {os.path.join(path, atk_id)}")


def save_tac_activations(atk_id, model_clean, model_bd, testset_clean, testset_bd, 
                        path, sample_size=None, batch_size=100):
    """
    Computes and saves Trigger-Activated Change (TAC) values.
    
    Args:
        atk_id (str): Attack identifier
        model_clean (torch.nn.Module): Clean model
        model_bd (torch.nn.Module): Backdoored model
        testset_clean (Dataset): Clean test dataset
        testset_bd (Dataset): Backdoored test dataset
        path (str): Directory to save TAC values
        sample_size (int, optional): Number of samples. Default: None (all)
        batch_size (int, optional): Batch size. Default: 100
        
    Saves:
        File '{path}/{atk_id}' containing:
        - 'clean': Dict with TAC from clean model
        - 'bd': Dict with TAC from backdoored model
        - 'indices': Sample indices
        
    Note:
        TAC is computed from a non-penultimate layer (penultimate=False)
    """
    sample_indices = np.array(range(len(testset_clean)))
    
    if sample_size:
        sample_indices = np.random.choice(sample_indices, size=sample_size, replace=False)
        testset_clean = torch.utils.data.Subset(testset_clean, sample_indices)
        testset_bd = torch.utils.data.Subset(testset_bd, sample_indices)

    test_features = {"indices": sample_indices}

    # Extract features from both clean and backdoored models
    for model, desc in zip([model_clean, model_bd], ["clean", "bd"]):
        preds_clean, features_clean = extract_preds_and_features(
            model, testset_clean, penultimate=False, batch_size=batch_size
        )
        preds_bd, features_bd = extract_preds_and_features(
            model, testset_bd, penultimate=False, batch_size=batch_size
        )

        # Compute activation difference
        diff = features_clean - features_bd

        # TAC = L2 norm over spatial dimensions (kernel height x width)
        tac_per_input = torch.norm(diff, dim=(2, 3))

        test_features[desc] = {
            "tac": tac_per_input,
            "predictions_clean": preds_clean,
            "predictions_bd": preds_bd,
        }

    os.makedirs(path, exist_ok=True)
    torch.save(test_features, os.path.join(path, atk_id))
    print(f"Saved TAC activations to {os.path.join(path, atk_id)}")


def save_tsne(atk_id, trainset, feature_train_path, path):
    """
    Creates and saves t-SNE embeddings and visualization.
    
    Args:
        atk_id (str): Attack identifier (e.g., 'badnet_p0.05')
        trainset (BackdoorDataset): Training dataset
        feature_train_path (str): Directory containing feature file
        path (str): Directory to save t-SNE results and plots
        
    Saves:
        - '{path}/{atk_id}/embedding.pt': t-SNE embeddings
        - '{path}/{atk_id}/*.png': Visualization plots
    """
    feature_file = os.path.join(feature_train_path, f"{atk_id}.pt")
    save_dir = os.path.join(path, atk_id)
    
    features_benign, features_poisoned, gt_labels_poisoned = create_tsne(
        trainset, feature_file, save_dst=save_dir
    )
    
    tsne = {
        "features_benign": features_benign,
        "features_poisoned": features_poisoned,
        "gt_labels_poisoned": gt_labels_poisoned
    }

    os.makedirs(save_dir, exist_ok=True)
    torch.save(tsne, os.path.join(save_dir, "embedding.pt"))
    print(f"Saved t-SNE to {os.path.join(save_dir, 'embedding.pt')}")


# =============================================================================
# SECTION 8: EVALUATION FUNCTIONS
# =============================================================================


def init_table(attacks, metrics):
    """
    Initializes a pandas DataFrame for storing evaluation results.
    
    Args:
        attacks (list): List of attack names (row indices)
        metrics (list): List of metric names (column indices)
        
    Returns:
        pd.DataFrame: Empty table with specified structure
    """
    table = pd.DataFrame(data=np.full((len(attacks), len(metrics)), None), 
                        index=attacks, columns=metrics)
    table.columns.name = "Attack"
    return table


def metric_over_batches(metric, dataset1, dataset2, batch_size=100):
    """
    Computes a metric over datasets in batches.
    
    Args:
        metric (callable): Metric function taking two image batches
        dataset1 (Dataset): First dataset
        dataset2 (Dataset): Second dataset
        batch_size (int, optional): Batch size. Default: 100
        
    Returns:
        float: Weighted average metric value
        
    Note:
        Uses weighted average to handle final batch correctly
    """
    dl1 = torch.utils.data.DataLoader(dataset1, batch_size=batch_size, shuffle=False)
    dl2 = torch.utils.data.DataLoader(dataset2, batch_size=batch_size, shuffle=False)
    metric_vals = []
    batch_weights = []
    
    for (in_batch1, _), (in_batch2, _) in zip(iter(dl1), iter(dl2)):
        assert len(in_batch1) == len(in_batch2)
        batch_weights.append(len(in_batch1))
        metric_vals.append(metric(in_batch1, in_batch2))

    return np.average(metric_vals, weights=batch_weights)


def input_baseline_eval(record_dict, dataset):
    """
    Evaluates baseline input-space similarity between consecutive clean images.
    
    This provides a reference for interpreting poisoned image similarity scores.
    
    Args:
        record_dict (dict): Record dictionary containing clean datasets
        dataset (str): Dataset name
        
    Returns:
        pd.DataFrame: Table with baseline values for all input-space metrics
        
    Example:
        >>> baseline_table = input_baseline_eval(record_dict, 'cifar10')
        >>> print(baseline_table)
    """
    METRIC_DICT = {
        "l1": l1_distance, "l2": l2_distance, "linf": linf_distance,
        "MSE": MSE, "PSNR": PSNR, "SSIM": SSIM,
        "LPIPS": LPIPS, "IS": IS, "pHash": pHash, "SAM": SAM,
    }

    input_baseline_table = init_table(["prototype"], METRIC_DICT.keys())
    clean_dataset = record_dict["prototype"]["test"]

    # Compare image i with image (i+1) mod N
    dataset_shifted = copy.deepcopy(clean_dataset)
    dataset_shifted.data = np.concatenate([
        np.expand_dims(clean_dataset.data[-1], axis=0),
        clean_dataset.data[:-1]
    ])

    for metric, metric_func in tqdm(METRIC_DICT.items(), 
                                   desc="Computing input-space baselines"):
        measurement = metric_over_batches(metric_func, clean_dataset, dataset_shifted)
        input_baseline_table.at["prototype", metric] = measurement

    return input_baseline_table


def input_stealth_eval(record_dict, pr_function, sample_size=None, split="test"):
    """
    Evaluates input-space stealthiness for backdoor attacks.
    
    Args:
        record_dict (dict): Nested dict of attack records
        pr_function (callable): Function mapping attack name to poison rate
        sample_size (int, optional): Number of samples to evaluate. Default: None (all)
        split (str, optional): 'train' or 'test'. Default: 'test'
        
    Returns:
        pd.DataFrame: Table with input-space metrics for each attack
        
    Example:
        >>> results = input_stealth_eval(
        ...     record_dict,
        ...     pr_function=lambda x: 0.05,
        ...     split='test'
        ... )
    """
    METRIC_DICT = {
        "l1": l1_distance, "l2": l2_distance, "linf": linf_distance,
        "MSE": MSE, "PSNR": PSNR, "SSIM": SSIM,
        "LPIPS": LPIPS, "IS": IS, "pHash": pHash, "SAM": SAM,
    }

    benign_and_attacks = list(record_dict.keys())
    attacks = [a for a in benign_and_attacks if a != "prototype"]
    input_stealth_table = init_table(benign_and_attacks, METRIC_DICT.keys())
    clean_dataset = record_dict["prototype"][split]

    # Determine poison indices
    if split == "train":
        # Verify all attacks have same poison indices
        atk_baseline = attacks[0]
        poison_lookup_ref = record_dict[atk_baseline][pr_function(atk_baseline)]["train"].poison_lookup
        
        for atk in attacks[1:]:
            pr = pr_function(atk)
            trainset = record_dict[atk][pr]["train"]
            if not np.all(trainset.poison_lookup == poison_lookup_ref):
                raise ValueError(f"Attack {atk} has different poison indices")
        
        poison_indices = np.argwhere(poison_lookup_ref).squeeze()
    else:
        # All test samples are poisoned
        poison_indices = range(len(clean_dataset))

    if sample_size:
        poison_indices = np.random.choice(poison_indices, size=sample_size, replace=False)
    
    clean_subset = torch.utils.data.Subset(clean_dataset, poison_indices)

    # Evaluate each attack
    for key in tqdm(benign_and_attacks, desc=f"Evaluating input-space ({split})"):
        if key == "prototype":
            bd_subset = clean_subset
        elif key == "dfba":
            bd_subset = torch.utils.data.Subset(record_dict[key][split], poison_indices)
        else:
            pr = pr_function(key)
            bd_subset = torch.utils.data.Subset(record_dict[key][pr][split], poison_indices)
            
        for metric, metric_func in METRIC_DICT.items():
            measurement = metric_over_batches(metric_func, clean_subset, bd_subset)
            input_stealth_table.at[key, metric] = measurement

    return input_stealth_table


def feature_stealth_eval(record_dict, model_arch, dataset, attacks, poison_rates,
                        result_dir="results", target_class=0):
    """
    Evaluates feature-space stealthiness metrics (SS and DSWD).
    
    Args:
        record_dict (dict): Nested dict of attack records
        model_arch (str): Model architecture
        dataset (str): Dataset name
        attacks (list): List of attack names
        poison_rates (list): List of poison rate functions
        result_dir (str, optional): Results directory. Default: 'results'
        target_class (int, optional): Target class. Default: 0
        
    Returns:
        pd.DataFrame: Table with SS and DSWD for each attack
        
    Example:
        >>> results = feature_stealth_eval(
        ...     record_dict,
        ...     model_arch='resnet18',
        ...     dataset='cifar10',
        ...     attacks=['badnet', 'blend'],
        ...     poison_rates=[lambda x: 0.05]
        ... )
    """
    n_classes_dict = {"cifar10": 10, "cifar100": 100, "imagenette": 10}
    n_classes = n_classes_dict[dataset]
    
    # Create attack-poison_rate combinations
    atk_pr_tuples = []
    for atk in attacks:
        if atk == "dfba":
            atk_pr_tuples.append((atk, None))
        else:
            for pr_func in poison_rates:
                atk_pr_tuples.append((atk, pr_func(atk)))
    
    rows = [f"{atk}_p{pr}" if pr else atk for atk, pr in atk_pr_tuples]
    metrics = ["SS", "DSWD"]
    feature_table = init_table(rows, metrics)
    
    exp_id = f"{model_arch}_{dataset}"
    
    for i, (atk, pr) in enumerate(tqdm(atk_pr_tuples, desc="Evaluating feature-space")):
        atk_id = atk if atk == "dfba" else f"{atk}_p{pr}"
        
        # Silhouette Score (skip for DFBA)
        if atk != "dfba":
            tsne_path = os.path.join(result_dir, "tsne", exp_id, atk_id, "embedding.pt")
            tsne_dict = torch.load(tsne_path, weights_only=False)
            feature_table.at[rows[i], "SS"] = SS(
                tsne_dict["features_benign"],
                tsne_dict["features_poisoned"]
            )
        
        # DSWD
        dict_key = record_dict[dataset][atk][pr] if pr else record_dict[dataset][atk]
        model = dict_key["model"]
        feature_path = os.path.join(result_dir, "feature_space_test", exp_id, f"{atk_id}.pt")
        feature_table.at[rows[i], "DSWD"] = DSWD(model, feature_path, 
                                                 model_arch=model_arch)
    
    return feature_table


def parameter_stealth_eval(record_dict, model_arch, dataset, attacks, poison_rates,
                          result_dir="results", target_class=0):
    """
    Evaluates parameter-space stealthiness metrics (UCLC and TAC).
    
    Args:
        record_dict (dict): Nested dict of attack records
        model_arch (str): Model architecture
        dataset (str): Dataset name
        attacks (list): List of attack names
        poison_rates (list): List of poison rate functions
        result_dir (str, optional): Results directory. Default: 'results'
        target_class (int, optional): Target class. Default: 0
        
    Returns:
        pd.DataFrame: Table with UCLC and TAC for each attack
        
    Example:
        >>> results = parameter_stealth_eval(
        ...     record_dict,
        ...     model_arch='resnet18',
        ...     dataset='cifar10',
        ...     attacks=['badnet', 'blend'],
        ...     poison_rates=[lambda x: 0.05]
        ... )
    """
    # Create attack-poison_rate combinations
    atk_pr_tuples = []
    for atk in attacks:
        if atk == "dfba":
            atk_pr_tuples.append((atk, None))
        else:
            for pr_func in poison_rates:
                atk_pr_tuples.append((atk, pr_func(atk)))
    
    rows = [f"{atk}_p{pr}" if pr else atk for atk, pr in atk_pr_tuples]
    metrics = ["UCLC", "TAC"]
    param_table = init_table(rows, metrics)
    
    exp_id = f"{model_arch}_{dataset}"
    
    for i, (atk, pr) in enumerate(tqdm(atk_pr_tuples, desc="Evaluating parameter-space")):
        atk_id = atk if atk == "dfba" else f"{atk}_p{pr}"
        dict_key = record_dict[dataset][atk][pr] if pr else record_dict[dataset][atk]
        model = dict_key["model"]
        
        # UCLC
        uclc = UCLC(model)
        param_table.at[rows[i], "UCLC"] = uclc.max().item()
        
        # TAC
        tac_path = os.path.join(result_dir, "tac_activations", exp_id, f"{atk_id}.pt")
        tac = TAC(tac_path, target_class=target_class)
        param_table.at[rows[i], "TAC"] = tac.max().item()
    
    return param_table


def new_metric_eval(record_dict, model_arch, dataset, attacks, poison_rates,
                   result_dir="results", target_class=0):
    """
    Evaluates novel metrics (CDBI and TUP) introduced in the paper.
    
    Args:
        record_dict (dict): Nested dict of attack records
        model_arch (str): Model architecture
        dataset (str): Dataset name
        attacks (list): List of attack names
        poison_rates (list): List of poison rate functions
        result_dir (str, optional): Results directory. Default: 'results'
        target_class (int, optional): Target class. Default: 0
        
    Returns:
        pd.DataFrame: Table with CDBI and TUP for each attack
        
    Example:
        >>> results = new_metric_eval(
        ...     record_dict,
        ...     model_arch='resnet18',
        ...     dataset='cifar10',
        ...     attacks=['badnet', 'blend'],
        ...     poison_rates=[lambda x: 0.05]
        ... )
    """
    n_classes_dict = {"cifar10": 10, "cifar100": 100, "imagenette": 10}
    n_classes = n_classes_dict[dataset]
    
    # Create attack-poison_rate combinations
    atk_pr_tuples = []
    for atk in attacks:
        if atk == "dfba":
            atk_pr_tuples.append((atk, None))
        else:
            for pr_func in poison_rates:
                atk_pr_tuples.append((atk, pr_func(atk)))
    
    rows = [f"{atk}_p{pr}" if pr else atk for atk, pr in atk_pr_tuples]
    metrics = ["CDBI", "TUP"]
    new_metric_table = init_table(rows, metrics)
    
    exp_id = f"{model_arch}_{dataset}"
    
    for i, (atk, pr) in enumerate(tqdm(atk_pr_tuples, desc="Evaluating new metrics")):
        atk_id = atk if atk == "dfba" else f"{atk}_p{pr}"
        dict_key = record_dict[dataset][atk][pr] if pr else record_dict[dataset][atk]
        
        # CDBI (skip for DFBA)
        if atk != "dfba":
            tsne_path = os.path.join(result_dir, "tsne", exp_id, atk_id, "embedding.pt")
            tsne_dict = torch.load(tsne_path, weights_only=False)
            mean_cdbi, _ = CDBI(
                tsne_dict["features_benign"],
                tsne_dict["features_poisoned"],
                tsne_dict["gt_labels_poisoned"],
                n_classes
            )
            new_metric_table.at[rows[i], "CDBI"] = mean_cdbi
        
        # TUP
        model = dict_key["model"]
        tac_path = os.path.join(result_dir, "tac_activations", exp_id, f"{atk_id}.pt")
        new_metric_table.at[rows[i], "TUP"] = TUP(tac_path, model, model_arch)
    
    return new_metric_table


def performance_eval(record_dict, model_arch, dataset, attacks, poison_rates,
                    result_dir="results", batch_size=100):
    """
    Evaluates attack performance: Benign Accuracy (BA) and Attack Success Rate (ASR).
    
    Args:
        record_dict (dict): Nested dict of attack records
        model_arch (str): Model architecture
        dataset (str): Dataset name
        attacks (list): List of attack names
        poison_rates (list): List of poison rate functions
        result_dir (str, optional): Results directory. Default: 'results'
        batch_size (int, optional): Batch size. Default: 100
        
    Returns:
        pd.DataFrame: Table with BA and ASR for each attack
        
    Example:
        >>> results = performance_eval(
        ...     record_dict,
        ...     model_arch='resnet18',
        ...     dataset='cifar10',
        ...     attacks=['badnet', 'blend'],
        ...     poison_rates=[lambda x: 0.05]
        ... )
    """
    # Create attack-poison_rate combinations (including prototype)
    benign_and_attacks = ["prototype"] + attacks
    atk_pr_tuples = []
    for atk in benign_and_attacks:
        if atk in ["prototype", "dfba"]:
            atk_pr_tuples.append((atk, None))
        else:
            for pr_func in poison_rates:
                atk_pr_tuples.append((atk, pr_func(atk)))
    
    rows = [f"{atk}_p{pr}" if pr else atk for atk, pr in atk_pr_tuples]
    metrics = ["BA", "ASR"]
    perf_table = init_table(rows, metrics)
    
    exp_id = f"{model_arch}_{dataset}"
    
    # Load full test dataset (with target class)
    testset_with_target = get_dataset(dataset, train=False)
    
    for i, (atk, pr) in enumerate(tqdm(atk_pr_tuples, desc="Evaluating performance")):
        atk_id = "prototype_pNone" if atk == "prototype" else (atk if atk == "dfba" else f"{atk}_p{pr}")
        dict_key = record_dict[dataset][atk]
        if pr is not None and atk not in ["prototype", "dfba"]:
            dict_key = dict_key[pr]
        
        model = dict_key["model"]
        
        # Benign Accuracy
        ba_preds_path = os.path.join(result_dir, "predictions_test_all_labels", 
                                    exp_id, f"{atk_id}.pt")
        
        if not os.path.exists(ba_preds_path):
            # Compute and save predictions
            preds = extract_preds(model, testset_with_target, batch_size=batch_size)
            ba_dict = {
                "predictions_clean": preds,
                "indices": np.arange(len(testset_with_target))
            }
            os.makedirs(os.path.dirname(ba_preds_path), exist_ok=True)
            torch.save(ba_dict, ba_preds_path)
        
        ba_dict = torch.load(ba_preds_path, weights_only=False)
        predictions = ba_dict["predictions_clean"].numpy()
        indices = ba_dict["indices"]
        labels = np.array(testset_with_target.targets)[indices]
        ba = np.mean(predictions == labels) * 100
        perf_table.at[rows[i], "BA"] = ba
        
        # Attack Success Rate (only for backdoored models)
        if atk != "prototype":
            testset_bd = dict_key["test"]
            preds_bd = extract_preds(model, testset_bd, batch_size=batch_size)
            asr = (preds_bd == 0).float().mean().item() * 100  # Assuming target_class=0
            perf_table.at[rows[i], "ASR"] = asr
    
    return perf_table


def results_to_csv(table, result_type, model_arch, dataset, result_dir="results"):
    """
    Saves evaluation results to CSV file.
    
    Args:
        table (pd.DataFrame): Results table
        result_type (str): Type of result (e.g., 'input_stealth', 'performance')
        model_arch (str): Model architecture
        dataset (str): Dataset name
        result_dir (str, optional): Results directory. Default: 'results'
        
    Saves:
        CSV file to: {result_dir}/tables/{result_type}_{model_arch}/{dataset}.csv
        
    Example:
        >>> results_to_csv(perf_table, 'performance', 'resnet18', 'cifar10')
    """
    path = os.path.join(result_dir, "tables", f"{result_type}_{model_arch}")
    os.makedirs(path, exist_ok=True)
    csv_path = os.path.join(path, f"{dataset}.csv")
    table.to_csv(csv_path)
    print(f"Results saved to {csv_path}")


def compare_poisoned_images(record_dict, pr_function, train=False, transformed=False,
                           index=0, fig_height=3, save_suffix="", save=True,
                           result_dir="results", model_arch="resnet18", dataset="cifar10"):
    """
    Creates visual comparison of poisoned images across attacks.
    
    Args:
        record_dict (dict): Nested dict of attack records
        pr_function (callable): Function mapping attack to poison rate
        train (bool, optional): If True, use train split; else test. Default: False
        transformed (bool, optional): If True, use transformed split. Default: False
        index (int, optional): Image index to visualize. Default: 0
        fig_height (float, optional): Figure height. Default: 3
        save_suffix (str, optional): Suffix for saved filenames. Default: ""
        save (bool, optional): If True, save individual images. Default: True
        result_dir (str, optional): Results directory. Default: 'results'
        model_arch (str): Model architecture
        dataset (str): Dataset name
        
    Saves:
        Individual poisoned images to:
        {result_dir}/poison_images_{split}/{model_arch}_{dataset}/{attack}{suffix}.png
        
    Example:
        >>> compare_poisoned_images(
        ...     record_dict,
        ...     pr_function=lambda x: 0.05,
        ...     train=False,
        ...     index=4,
        ...     save=True
        ... )
    """
    benign_and_attacks = list(record_dict.keys())
    fig, ax = plt.subplots(1, len(benign_and_attacks))
    split = "train" if train else "test"
    
    if save:
        save_path = os.path.join(result_dir, f"poison_images_{split}", 
                                f"{model_arch}_{dataset}")
        os.makedirs(save_path, exist_ok=True)
    
    split += "_transformed" if transformed else ""

    for i, key in enumerate(benign_and_attacks):
        if key in ["prototype", "dfba"]:
            dataset_obj = record_dict[key][split]
        else:
            pr = pr_function(key)
            dataset_obj = record_dict[key][pr][split]

        img, _ = dataset_obj.__getitem__(index)
        ax[i].imshow(img.permute(1, 2, 0))
        ax[i].set_title(key)
        ax[i].axis("off")

        if save:
            to_pil = T.ToPILImage()
            img_pil = to_pil(img)
            img_pil.save(os.path.join(save_path, f"{key}{save_suffix}.png"))

    fig.suptitle("Comparison of poisoned images for different attacks")
    fig.set_figheight(fig_height)
    fig.set_figwidth(fig_height * len(benign_and_attacks))
    plt.tight_layout()
    plt.show()


# =============================================================================
# MODULE METADATA
# =============================================================================

__version__ = "1.0.0"
__author__ = "Research Team"
__all__ = [
    # Dataset classes
    "Imagenette", "BackdoorDataset", "BackdoorBenchDataset",
    "AdapDataset", "DFSTDataset", "DFBADataset", "GrondDataset",
    
    # Utility functions
    "filter_target_class", "extract_preds", "extract_preds_and_features",
    "experiment_variable_identifier", "list_intersection", "dict_subset",
    
    # Input-space metrics
    "l1_distance", "l2_distance", "linf_distance", "MSE", "PSNR", "SSIM",
    "pHash", "LPIPS", "IS", "SAM",
    
    # Feature-space metrics
    "create_tsne", "clustering_score", "SS", "CDBI", "DSWD", "DSWD_eq_7",
    
    # Parameter-space metrics
    "UCLC", "TAC", "TAC_comparison", "TUP",
    
    # Data and model loading
    "get_dataset", "load_model_state", "load_clean_record", "load_backdoor_record",
    "load_backdoorbench", "load_adap", "load_dfst", "load_dfba", "load_grond",
    
    # Feature extraction
    "get_sample_size", "save_train_feature_space", "save_test_feature_space",
    "save_tac_activations", "save_tsne",
    
    # Evaluation functions
    "init_table", "metric_over_batches", "input_baseline_eval", "input_stealth_eval",
    "feature_stealth_eval", "parameter_stealth_eval", "new_metric_eval",
    "performance_eval", "results_to_csv", "compare_poisoned_images",
]


if __name__ == "__main__":
    print(f"Backdoor Stealthiness Evaluation Utilities v{__version__}")
    print(f"Loaded {len(__all__)} exported symbols")
    print("\nExample usage:")
    print("  from eval_utils import BackdoorDataset, LPIPS, extract_preds")
    print("  # See function docstrings for detailed usage")
