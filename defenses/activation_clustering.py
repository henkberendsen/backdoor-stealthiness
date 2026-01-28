#!/usr/bin/env python3
"""
Activation Clustering: Detecting Backdoor Attacks on Deep Neural Networks

Implementation of Activation Clustering defense for backdoor detection.
The defense works by clustering activation patterns of training samples per class
and identifying outlier clusters that may contain poisoned samples.

Reference:
    Chen et al. "Detecting backdoor attacks on deep neural networks by activation clustering"
    arXiv preprint arXiv:1811.03728, 2018

Usage:
    python defenses/activation_clustering.py --attack badnet --model resnet18 --dataset cifar10 --poison_rate 0.05 --exp_num 1
"""

import argparse
import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import random
import csv
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, silhouette_score
from sklearn.cluster import KMeans
from sklearn.decomposition import FastICA
from tqdm import tqdm
import copy
import torchvision.transforms.v2 as T

# Add project root to path
REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(REPO_ROOT))

# Import utilities from eval_utils
from eval_utils import (
    BackdoorBenchDataset,
    load_model_state,
    get_dataset,
    load_backdoor_record,
    load_clean_record,
    experiment_variable_identifier,
    detect_image_size_from_attack_path,
    IMG_SIZE_DICT
)

# Constants
RECORD_DIR = REPO_ROOT / "large_files" / "record"
RESULTS_DIR = REPO_ROOT / "defenses" / "results"
DATA_DIR = REPO_ROOT / "large_files" / "data"
TARGET_CLASS = 0


def set_random_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_features(model, model_name, dataloader, device):
    """
    Extract features/activations from a specific layer of the model.
    
    Args:
        model: The neural network model
        model_name: Name of the model architecture
        dataloader: DataLoader for the dataset
        device: Device to run on
    
    Returns:
        torch.Tensor: Extracted features for all samples
    """
    model.eval()
    activations_all = []
    
    with torch.no_grad():
        for x_batch, _ in tqdm(dataloader, desc="  Extracting features", leave=False):
            x_batch = x_batch.to(device)
            
            # Register hook based on model architecture
            outs = []
            inps = []
            
            def layer_hook(module, inp, out):
                outs.append(out.data)
            
            def layer_hook_inp(module, inp, out):
                inps.append(inp[0].data)
            
            # Select appropriate layer based on architecture
            if model_name == 'resnet18':
                hook = model.layer4.register_forward_hook(layer_hook)
            elif model_name in ['vgg19', 'vgg19_bn']:
                hook = model.features.register_forward_hook(layer_hook)
            elif model_name == 'preactresnet18':
                hook = model.avgpool.register_forward_hook(layer_hook)
            elif model_name in ['mobilenet_v3_large', 'efficientnet_b3', 'convnext_tiny']:
                hook = model.avgpool.register_forward_hook(layer_hook)
            elif model_name == 'densenet161':
                hook = model.features.register_forward_hook(layer_hook)
            elif model_name == 'vit_b_16':
                hook = model[1].heads.register_forward_hook(layer_hook_inp)
            else:
                raise ValueError(f"Unsupported model architecture: {model_name}")
            
            # Forward pass
            _ = model(x_batch)
            
            # Extract activations
            if model_name == 'vit_b_16':
                activations = inps[0].view(inps[0].size(0), -1)
            elif model_name == 'densenet161':
                activations = torch.nn.functional.relu(outs[0])
                activations = activations.view(activations.size(0), -1)
            else:
                activations = outs[0].view(outs[0].size(0), -1)
            
            activations_all.append(activations.cpu())
            hook.remove()
    
    activations_all = torch.cat(activations_all, axis=0)
    return activations_all


class ActivationClusteringDetector:
    """
    Activation Clustering detector for backdoor detection.
    """
    
    def __init__(self, model, model_name, device, num_classes, 
                 nb_clusters=2, nb_dims=10, random_seed=42):
        """
        Initialize the Activation Clustering detector.
        
        Args:
            model: The backdoored model
            model_name: Model architecture name
            device: Device to run on
            num_classes: Number of classes in the dataset
            nb_clusters: Number of clusters for K-means (default: 2)
            nb_dims: Number of dimensions for ICA reduction (default: 10)
            random_seed: Random seed for reproducibility
        """
        self.model = model
        self.model_name = model_name
        self.device = device
        self.num_classes = num_classes
        self.nb_clusters = nb_clusters
        self.nb_dims = nb_dims
        self.random_seed = random_seed
    
    def detect(self, train_loader, train_labels):
        """
        Detect poisoned samples in the training set using activation clustering.
        
        Args:
            train_loader: DataLoader for training data
            train_labels: Labels for training data
        
        Returns:
            list: Indices of suspected poisoned samples
        """
        print(f"  Extracting features from model...")
        
        # Extract features
        features = get_features(self.model, self.model_name, train_loader, self.device)
        
        print(f"  Performing per-class clustering...")
        
        # Get class indices
        class_indices = []
        train_labels_array = np.array(train_labels)
        for i in range(self.num_classes):
            idx = np.where(train_labels_array == i)[0]
            class_indices.append(idx)
        
        suspicious_indices = []
        
        # Perform clustering per class
        for target_class in range(self.num_classes):
            print(f"    - Processing class {target_class}...")
            
            if len(class_indices[target_class]) <= 1:
                print(f"      Warning: Skipping (insufficient samples)")
                continue
            
            # Get features for this class
            temp_feats = features[class_indices[target_class]]
            
            # Center the features
            temp_feats = temp_feats - temp_feats.mean(dim=0)
            
            # Dimensionality reduction with FastICA
            X = temp_feats.cpu().numpy()
            n_components = min(self.nb_dims, X.shape[0], X.shape[1])
            
            if n_components < 2:
                print(f"      Warning: Skipping (insufficient dimensions)")
                continue
            
            try:
                transformer = FastICA(
                    n_components=n_components,
                    random_state=self.random_seed,
                    whiten='unit-variance',
                    max_iter=1000
                )
                X_transformed = transformer.fit_transform(X)
            except Exception as e:
                print(f"      Warning: FastICA failed: {e}, using raw features")
                X_transformed = X
            
            # K-means clustering
            kmeans = KMeans(n_clusters=self.nb_clusters, random_state=self.random_seed).fit(X_transformed)
            
            # Identify clean cluster (majority cluster)
            if kmeans.labels_.sum() >= len(kmeans.labels_) / 2.:
                clean_label = 1
            else:
                clean_label = 0
            
            # Find outliers
            outliers = []
            for idx, label in enumerate(kmeans.labels_):
                if label != clean_label:
                    outliers.append(class_indices[target_class][idx])
            
            # Calculate silhouette score
            if len(np.unique(kmeans.labels_)) > 1:
                score = silhouette_score(X_transformed, kmeans.labels_)
                print(f"      Silhouette score: {score:.4f}, Outliers: {len(outliers)}")
                
                # Only add outliers if they're not too many (< 35% of class samples)
                if len(outliers) < len(kmeans.labels_) * 0.35:
                    suspicious_indices.extend(outliers)
                else:
                    print(f"      Warning: Too many outliers ({len(outliers)}/{len(kmeans.labels_)}), skipping")
            else:
                print(f"      Warning: Only one cluster found, skipping")
        
        return suspicious_indices


class TransformedDataset(torch.utils.data.Dataset):
    """Wrapper to apply transforms to a dataset."""
    
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def verify_attack_exists(attack_name, model_arch, dataset, poison_rate):
    """
    Verify that the attack directory exists.
    
    Returns:
        Path to the attack directory
    """
    # Convert poison_rate to the format used in directory names
    pr_str = str(poison_rate).replace(".", "-")
    pattern = f"{attack_name}_{model_arch}_{dataset}_p{pr_str}"
    
    attack_dir = RECORD_DIR / pattern
    
    if not attack_dir.exists():
        raise FileNotFoundError(
            f"Attack directory not found: {attack_dir}\n"
            f"Expected pattern: {pattern}\n"
            f"Make sure the attack has been trained and saved in {RECORD_DIR}"
        )
    
    return attack_dir


def load_attack_data(attack_name, dataset_name, model_arch, poison_rate, device):
    """
    Load backdoored model and datasets for any attack type.
    
    Uses the proper loading functions from eval_utils.py to handle all attack types.
    
    Args:
        attack_name: Name of the attack
        dataset_name: Dataset name (cifar10, cifar100, etc.)
        model_arch: Model architecture (resnet18, vgg16, etc.)
        poison_rate: Poison rate (e.g., 0.05)
        device: torch device
    
    Returns:
        dict with keys: 'model', 'bd_train', 'bd_test', 'clean_train', 'clean_test'
    """
    print(f"  Loading attack: {attack_name}")
    
    # Build attack path to detect image size (for imagenette compatibility)
    exp_id = experiment_variable_identifier(model_arch, dataset_name, poison_rate)
    attack_path = str(RECORD_DIR / f"{attack_name}_{exp_id}")
    
    # Detect image size from poisoned data (important for imagenette)
    detected_img_size = detect_image_size_from_attack_path(attack_path)
    img_size = detected_img_size if detected_img_size else IMG_SIZE_DICT.get(dataset_name)
    if img_size:
        print(f"  Detected image size: {img_size}x{img_size}")
    
    # Load clean record first
    print(f"  Loading clean datasets and model...")
    clean_record = load_clean_record(
        dataset=dataset_name,
        arch=model_arch,
        record_dir=str(RECORD_DIR),
        data_dir=str(DATA_DIR),
        target_class=TARGET_CLASS,
        img_size=img_size
    )
    
    # Load backdoor record using eval_utils dispatcher
    print(f"  Loading backdoored model and datasets...")
    bd_record = load_backdoor_record(
        dataset=dataset_name,
        arch=model_arch,
        atk=attack_name,
        poison_rate=poison_rate,
        clean_record=clean_record,
        record_dir=str(RECORD_DIR)
    )
    
    # Get the model
    model = bd_record['model']
    model.to(device)
    model.eval()
    
    # Get datasets - try 'train' first, fallback to 'train_transformed' for DFST/Grond
    bd_train = bd_record.get('train', None)
    if bd_train is None:
        bd_train = bd_record.get('train_transformed', None)
        if bd_train is not None:
            print(f"  Note: Using 'train_transformed' for {attack_name} (no untransformed train set available)")
    
    if bd_train is None:
        # Some attacks like DFBA are data-free and don't have training data
        raise ValueError(
            f"No training dataset found for attack {attack_name}\n"
            f"Available keys in bd_record: {list(bd_record.keys())}\n"
            f"Note: Activation Clustering requires training data. "
            f"This attack may not be compatible with AC defense."
        )
    
    bd_test = bd_record.get('test', None)
    
    # Use clean datasets from clean_record
    clean_train = clean_record['train']
    clean_test = clean_record['test']
    
    print(f"  Loaded successfully")
    print(f"    - Model: {model_arch}")
    print(f"    - Backdoor train samples: {len(bd_train)}")
    if bd_test is not None:
        print(f"    - Backdoor test samples: {len(bd_test)}")
    print(f"    - Clean train samples: {len(clean_train)}")
    print(f"    - Clean test samples: {len(clean_test)}")
    
    return {
        'model': model,
        'bd_train': bd_train,
        'bd_test': bd_test,
        'clean_train': clean_train,
        'clean_test': clean_test,
        'attack_record': bd_record,
        'clean_record': clean_record
    }


def prepare_train_data(bd_train, batch_size=128):
    """
    Prepare training data for feature extraction.
    
    Args:
        bd_train: Backdoored training dataset
        batch_size: Batch size for DataLoader
    
    Returns:
        tuple: (train_loader, train_labels, poison_indices)
    """
    print(f"  Preparing training data...")
    
    # Create transform to ensure tensors
    to_tensor_transform = T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True)
    ])
    
    # Collect all samples and labels
    train_samples = []
    train_labels = []
    
    for i in range(len(bd_train)):
        img, label = bd_train[i]
        # Convert PIL Image to tensor if needed
        if not isinstance(img, torch.Tensor):
            img = to_tensor_transform(img)
        train_samples.append(img)
        train_labels.append(label)
    
    # Get poison indices (ground truth)
    poison_indices = []
    if hasattr(bd_train, 'poison_lookup') and bd_train.poison_lookup is not None:
        for i in range(len(bd_train)):
            if i < len(bd_train.poison_lookup) and bd_train.poison_lookup[i]:
                poison_indices.append(i)
    
    print(f"    - Total training samples: {len(train_samples)}")
    print(f"    - Poisoned samples (ground truth): {len(poison_indices)}")
    
    # Create dataset
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, samples, labels):
            self.samples = samples
            self.labels = labels
        
        def __len__(self):
            return len(self.samples)
        
        def __getitem__(self, idx):
            return self.samples[idx], self.labels[idx]
    
    train_dataset = SimpleDataset(train_samples, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, train_labels, poison_indices


def calculate_metrics(ground_truth, predictions):
    """
    Calculate detection metrics.
    
    Args:
        ground_truth: Array of ground truth labels (1 for poisoned, 0 for clean)
        predictions: List of indices predicted as poisoned
    
    Returns:
        dict with TPR, FPR, precision, recall, accuracy
    """
    if len(predictions) == 0:
        # No detections made
        tn = np.sum(ground_truth == 0)
        fp = 0
        fn = np.sum(ground_truth == 1)
        tp = 0
    else:
        # Create prediction array
        pred_array = np.zeros(len(ground_truth))
        pred_array[predictions] = 1
        
        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(ground_truth, pred_array).ravel()
    
    # Calculate metrics
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tpr
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'TP': int(tp),
        'TN': int(tn),
        'FP': int(fp),
        'FN': int(fn),
        'TPR': float(tpr),
        'FPR': float(fpr),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'accuracy': float(accuracy),
        'detected_poisoned': int(tp + fp),  # Total flagged as poisoned
        'detected_clean': int(tn + fn)      # Total identified as clean
    }


def save_results_to_csv(results, csv_path):
    """Save results to CSV file."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if file exists to determine if we need headers
    file_exists = csv_path.exists()
    
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'exp_num', 'defense_name', 'attack', 'model', 'dataset', 
            'poison_rate', 'FPR', 'TPR', 'precision', 'recall', 'f1',
            'detected_poisoned', 'detected_clean', 'TP', 'TN', 'FP', 'FN', 'accuracy'
        ])
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(results)
    
    print(f"\nResults saved to: {csv_path}")


def run_activation_clustering_defense(attack_name, model_arch, dataset_name, poison_rate,
                                      nb_clusters=2, nb_dims=10, batch_size=128, 
                                      random_seed=42, exp_num=None):
    """
    Run Activation Clustering defense on a backdoor attack.
    
    Args:
        attack_name: Name of the attack
        model_arch: Model architecture
        dataset_name: Dataset name
        poison_rate: Poison rate
        nb_clusters: Number of clusters for K-means
        nb_dims: Number of dimensions for ICA reduction
        batch_size: Batch size for processing
        random_seed: Random seed for reproducibility
        exp_num: Experiment number (should be provided by user)
    """
    print("=" * 70)
    print("Activation Clustering Backdoor Defense")
    print("=" * 70)
    print(f"Attack: {attack_name}")
    print(f"Model: {model_arch}")
    print(f"Dataset: {dataset_name}")
    print(f"Poison Rate: {poison_rate}")
    print(f"AC Parameters: nb_clusters={nb_clusters}, nb_dims={nb_dims}")
    print("=" * 70)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    # Set random seed
    set_random_seed(random_seed)
    
    # Verify attack exists
    print(f"\nVerifying attack directory...")
    attack_dir = verify_attack_exists(attack_name, model_arch, dataset_name, poison_rate)
    print(f"  Found: {attack_dir}")
    
    # Load attack data (handles all attack types)
    print(f"\nLoading attack data...")
    attack_data = load_attack_data(attack_name, dataset_name, model_arch, poison_rate, device)
    model = attack_data['model']
    bd_train = attack_data['bd_train']
    
    # Get number of classes
    if dataset_name == 'cifar10':
        num_classes = 10
    elif dataset_name == 'cifar100':
        num_classes = 100
    elif dataset_name == 'imagenette':
        num_classes = 10
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    # Prepare training data
    print(f"\nPreparing training data...")
    train_loader, train_labels, poison_indices = prepare_train_data(bd_train, batch_size)
    
    # Create ground truth array
    ground_truth = np.zeros(len(train_labels))
    for idx in poison_indices:
        ground_truth[idx] = 1
    
    # Initialize detector
    print(f"\nInitializing Activation Clustering detector...")
    detector = ActivationClusteringDetector(
        model=model,
        model_name=model_arch,
        device=device,
        num_classes=num_classes,
        nb_clusters=nb_clusters,
        nb_dims=nb_dims,
        random_seed=random_seed
    )
    
    # Run detection
    print(f"\nRunning Activation Clustering detection...")
    suspicious_indices = detector.detect(train_loader, train_labels)
    
    print(f"\n  Detection complete")
    print(f"  Flagged as suspicious: {len(suspicious_indices)}/{len(train_labels)} samples")
    
    # Calculate metrics
    print(f"\nCalculating metrics...")
    metrics = calculate_metrics(ground_truth, suspicious_indices)
    
    # Print results
    print(f"\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"True Positives (TP):  {metrics['TP']:6d}  (Correctly detected poisoned)")
    print(f"True Negatives (TN):  {metrics['TN']:6d}  (Correctly identified clean)")
    print(f"False Positives (FP): {metrics['FP']:6d}  (Clean flagged as poisoned)")
    print(f"False Negatives (FN): {metrics['FN']:6d}  (Poisoned missed)")
    print("-" * 70)
    print(f"True Positive Rate (TPR/Recall): {metrics['TPR']:.4f}")
    print(f"False Positive Rate (FPR):       {metrics['FPR']:.4f}")
    print(f"Precision:                        {metrics['precision']:.4f}")
    print(f"F1 Score:                         {metrics['f1']:.4f}")
    print(f"Accuracy:                         {metrics['accuracy']:.4f}")
    print("=" * 70)
    
    # Prepare results for CSV
    csv_path = RESULTS_DIR / "activation_clustering.csv"
    
    results = {
        'exp_num': exp_num,
        'defense_name': 'ActivationClustering',
        'attack': attack_name,
        'model': model_arch,
        'dataset': dataset_name,
        'poison_rate': poison_rate,
        'FPR': metrics['FPR'],
        'TPR': metrics['TPR'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1': metrics['f1'],
        'detected_poisoned': metrics['detected_poisoned'],
        'detected_clean': metrics['detected_clean'],
        'TP': metrics['TP'],
        'TN': metrics['TN'],
        'FP': metrics['FP'],
        'FN': metrics['FN'],
        'accuracy': metrics['accuracy']
    }
    
    # Save results
    save_results_to_csv(results, csv_path)
    
    return results


def main():
    """Main function to run the defense."""
    parser = argparse.ArgumentParser(
        description='Activation Clustering Defense Against Backdoor Attacks',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--attack', type=str, required=True,
                       help='Attack name (e.g., badnet, wanet, adaptive_blend)')
    parser.add_argument('--model', type=str, required=True,
                       help='Model architecture (e.g., resnet18)')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name (cifar10, cifar100, imagenette)')
    parser.add_argument('--poison_rate', type=float, required=True,
                       help='Poison rate (e.g., 0.05, 0.003)')
    parser.add_argument('--exp_num', type=int, default=0,
                       help='Experiment number for tracking')
    
    # AC parameters
    parser.add_argument('--nb_clusters', type=int, default=2,
                       help='Number of clusters for K-means')
    parser.add_argument('--nb_dims', type=int, default=10,
                       help='Number of dimensions for ICA reduction')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size for processing')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    try:
        print("Starting Activation Clustering Defense")
        run_activation_clustering_defense(
            attack_name=args.attack,
            model_arch=args.model,
            dataset_name=args.dataset,
            poison_rate=args.poison_rate,
            nb_clusters=args.nb_clusters,
            nb_dims=args.nb_dims,
            batch_size=args.batch_size,
            random_seed=args.random_seed,
            exp_num=args.exp_num
        )
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
