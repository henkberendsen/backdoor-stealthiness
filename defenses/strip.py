#!/usr/bin/env python3
"""
STRIP: A Defence Against Trojan Attacks on Deep Neural Networks

Implementation of STRIP defense for backdoor detection.
The defense works by mixing test samples with clean samples and measuring
the entropy of predictions. Poisoned samples exhibit lower entropy variance.

Reference:
    Gao et al. "STRIP: A Defence Against Trojan Attacks on Deep Neural Networks"
    ACSAC 2019

Usage:
    python defenses/strip.py --attack badnet --model resnet18 --dataset cifar10 --poison_rate 0.05 --exp_num 1
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
from sklearn.metrics import confusion_matrix
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

# Attack type classification
BACKDOORBENCH_ATTACKS = ["badnet", "blended", "wanet", "bpp", "narcissus"]
ADAPTIVE_ATTACKS = ["adaptive_patch", "adaptive_blend"]
OTHER_ATTACKS = ["dfst", "grond", "dfba"]
ALL_ATTACKS = BACKDOORBENCH_ATTACKS + ADAPTIVE_ATTACKS + OTHER_ATTACKS


class TransformedDataset(torch.utils.data.Dataset):
    """
    Wrapper to apply transforms to a dataset that may not have them.
    """
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        
        # Convert PIL Image to tensor if needed
        if self.transform is not None:
            img = self.transform(img)
        
        return img, label


class STRIPDetector:
    """
    STRIP detector implementation.
    
    Args:
        model: Neural network model
        clean_set: Dataset of clean samples for mixing
        device: torch device
        strip_alpha: Blending coefficient (default: 1.0)
        N: Number of clean samples to mix with each test sample (default: 100)
        defense_fpr: Desired false positive rate for threshold (default: 0.1)
    """
    
    def __init__(self, model, clean_set, device, strip_alpha=1.0, N=100, defense_fpr=0.1):
        self.model = model
        self.clean_set = clean_set
        self.device = device
        self.strip_alpha = strip_alpha
        self.N = N
        self.defense_fpr = defense_fpr
        self.threshold_low = None
        self.threshold_high = None
        
    def superimpose(self, input1, input2, alpha=None):
        """Blend two images together."""
        if alpha is None:
            alpha = self.strip_alpha
        return input1 + alpha * input2
    
    def entropy(self, inputs):
        """Calculate entropy of model predictions."""
        with torch.no_grad():
            p = torch.nn.functional.softmax(self.model(inputs), dim=1) + 1e-8
            return (-p * p.log()).sum(1)
    
    def check_sample(self, input_tensor):
        """
        Check a single sample by mixing with N clean samples and calculating average entropy.
        
        Args:
            input_tensor: Input tensor (can be a batch)
            
        Returns:
            Average entropy across N mixtures
        """
        entropy_list = []
        
        # Sample N random clean samples
        samples = list(range(len(self.clean_set)))
        random.shuffle(samples)
        samples = samples[:self.N]
        
        with torch.no_grad():
            for idx in samples:
                clean_img, _ = self.clean_set[idx]
                clean_img = clean_img.to(self.device)
                
                # Superimpose the input with clean sample
                mixed = self.superimpose(input_tensor, clean_img)
                
                # Calculate entropy
                entropy = self.entropy(mixed).cpu()
                entropy_list.append(entropy)
        
        # Return mean entropy across all mixtures
        return torch.stack(entropy_list).mean(0)
    
    def calibrate_threshold(self, calibration_set_loader):
        """
        Calibrate detection threshold using clean samples.
        
        Args:
            calibration_set_loader: DataLoader with clean samples
        """
        print("  Calibrating thresholds with clean samples...")
        clean_entropies = []
        
        for inputs, _ in tqdm(calibration_set_loader, desc="  Calibrating", leave=False):
            inputs = inputs.to(self.device)
            entropies = self.check_sample(inputs)
            
            # Handle both single values and batches
            if entropies.dim() == 0:
                clean_entropies.append(entropies.item())
            else:
                clean_entropies.extend(entropies.tolist())
        
        clean_entropies = torch.FloatTensor(clean_entropies)
        clean_entropies, _ = clean_entropies.sort()
        
        # Set threshold at the specified FPR
        self.threshold_low = float(clean_entropies[int(self.defense_fpr * len(clean_entropies))])
        self.threshold_high = np.inf
        
        print(f"  Threshold calibrated: {self.threshold_low:.4f}")
    
    def detect(self, test_loader):
        """
        Detect poisoned samples in test set.
        
        Args:
            test_loader: DataLoader with samples to test
            
        Returns:
            numpy array of indices flagged as suspicious
        """
        print("  Running STRIP detection...")
        all_entropies = []
        
        for inputs, _ in tqdm(test_loader, desc="  Detecting", leave=False):
            inputs = inputs.to(self.device)
            entropies = self.check_sample(inputs)
            
            # Handle both single values and batches
            if entropies.dim() == 0:
                all_entropies.append(entropies.item())
            else:
                all_entropies.extend(entropies.tolist())
        
        all_entropies = torch.FloatTensor(all_entropies)
        
        # Flag samples outside threshold range
        suspicious = torch.logical_or(
            all_entropies < self.threshold_low,
            all_entropies > self.threshold_high
        ).nonzero().reshape(-1)
        
        return suspicious.numpy()


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
        dict with 'model', 'bd_test', 'clean_test', 'attack_record'
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
    
    # Load clean record first (needed for some attacks)
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
    
    # Get test datasets
    bd_test = bd_record.get('test', None)
    if bd_test is None:
        raise ValueError(f"No test dataset found for attack {attack_name}")
    
    # Use clean test from clean_record (filtered for target class)
    clean_test = clean_record['test']
    
    print(f"  Loaded successfully")
    print(f"    - Model: {model_arch}")
    print(f"    - Backdoor test samples: {len(bd_test)}")
    print(f"    - Clean test samples: {len(clean_test)}")
    
    return {
        'model': model,
        'bd_test': bd_test,
        'clean_test': clean_test,
        'attack_record': bd_record,
        'clean_record': clean_record
    }


def prepare_test_samples(bd_test, clean_test, n_poisoned=None, n_clean=1000):
    """
    Prepare test samples: mix poisoned samples with clean samples.
    
    Args:
        bd_test: Backdoored test dataset
        clean_test: Clean test dataset  
        n_poisoned: Number of poisoned samples (None = all)
        n_clean: Number of clean samples
        
    Returns:
        tuple: (test_samples, test_labels, ground_truth_labels)
    """
    print(f"  Preparing test samples...")
    
    # Create transform for clean images (bd_test already returns tensors)
    to_tensor_transform = T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True)
    ])
    
    # Get poisoned samples - iterate through actual available indices
    # poison_lookup may have more entries than the dataset (due to filtering)
    # so we only check indices that actually exist in bd_test
    poison_indices = []
    for i in range(len(bd_test)):
        if i < len(bd_test.poison_lookup) and bd_test.poison_lookup[i]:
            poison_indices.append(i)
    
    poison_indices = np.array(poison_indices)
    
    if n_poisoned is not None and n_poisoned < len(poison_indices):
        poison_indices = np.random.choice(poison_indices, n_poisoned, replace=False)
    
    print(f"    - Poisoned samples: {len(poison_indices)}")
    print(f"    - Dataset size: {len(bd_test)}")
    
    # Get random clean samples
    clean_indices = np.random.choice(len(clean_test), 
                                     min(n_clean, len(clean_test)), 
                                     replace=False)
    print(f"    - Clean samples: {len(clean_indices)}")
    
    # Collect samples
    test_samples = []
    test_labels = []
    ground_truth = []  # 1 for poisoned, 0 for clean
    
    # Add poisoned samples (convert to tensors if needed)
    for idx in poison_indices:
        img, label = bd_test[idx]
        # Convert PIL Image to tensor if needed (adaptive attacks return PIL Images)
        if not isinstance(img, torch.Tensor):
            img = to_tensor_transform(img)
        test_samples.append(img)
        test_labels.append(label)
        ground_truth.append(1)
    
    # Add clean samples (need to convert to tensors)
    for idx in clean_indices:
        img, label = clean_test[idx]
        # Convert PIL Image to tensor if needed
        if not isinstance(img, torch.Tensor):
            img = to_tensor_transform(img)
        test_samples.append(img)
        test_labels.append(label)
        ground_truth.append(0)
    
    return test_samples, test_labels, np.array(ground_truth)


def calculate_metrics(ground_truth, predictions):
    """
    Calculate detection metrics.
    
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


def run_strip_defense(attack_name, model_arch, dataset_name, poison_rate, 
                      strip_alpha=1.0, N=100, defense_fpr=0.1, 
                      n_clean_calibration=500, batch_size=128, exp_num=None):
    """
    Run STRIP defense on a backdoor attack.
    
    Args:
        attack_name: Name of the attack
        model_arch: Model architecture
        dataset_name: Dataset name
        poison_rate: Poison rate
        strip_alpha: STRIP blending coefficient
        N: Number of clean samples to mix
        defense_fpr: Desired FPR for threshold
        n_clean_calibration: Number of clean samples for calibration
        batch_size: Batch size for processing
        exp_num: Experiment number (should be provided by user)
    """
    print("=" * 70)
    print("STRIP Backdoor Defense")
    print("=" * 70)
    print(f"Attack: {attack_name}")
    print(f"Model: {model_arch}")
    print(f"Dataset: {dataset_name}")
    print(f"Poison Rate: {poison_rate}")
    print(f"STRIP Parameters: alpha={strip_alpha}, N={N}, FPR={defense_fpr}")
    print("=" * 70)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    # Verify attack exists
    print(f"\nVerifying attack directory...")
    attack_dir = verify_attack_exists(attack_name, model_arch, dataset_name, poison_rate)
    print(f"  Found: {attack_dir}")
    
    # Load attack data (handles all attack types)
    print(f"\nLoading attack data...")
    attack_data = load_attack_data(attack_name, dataset_name, model_arch, poison_rate, device)
    model = attack_data['model']
    bd_test = attack_data['bd_test']
    clean_test = attack_data['clean_test']
    attack_record = attack_data['attack_record']
    clean_record = attack_data['clean_record']
    
    # Prepare clean samples for STRIP mixing
    print(f"\nPreparing clean reference set...")
    # Get the underlying dataset from clean_test (which is already filtered for target class)
    # clean_test is a regular dataset (CIFAR10/100 or Imagenette), not wrapped
    
    # Create transform to convert PIL Images to tensors
    to_tensor_transform = T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True)
    ])
    
    # Wrap clean_test with transforms
    clean_test_transformed = TransformedDataset(clean_test, transform=to_tensor_transform)
    
    clean_indices = np.random.choice(len(clean_test_transformed), 
                                     min(n_clean_calibration, len(clean_test_transformed)), 
                                     replace=False)
    clean_reference = torch.utils.data.Subset(clean_test_transformed, clean_indices)
    print(f"  Clean reference set: {len(clean_reference)} samples")
    
    # Initialize STRIP detector
    print(f"\nInitializing STRIP detector...")
    detector = STRIPDetector(model, clean_reference, device, 
                            strip_alpha=strip_alpha, N=N, defense_fpr=defense_fpr)
    
    # Calibrate threshold with clean samples
    calibration_loader = DataLoader(clean_reference, batch_size=batch_size, shuffle=False)
    detector.calibrate_threshold(calibration_loader)
    
    # Prepare test samples (mix poisoned and clean)
    test_samples, test_labels, ground_truth = prepare_test_samples(bd_test, clean_test)
    
    # Create dataset for testing
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, samples, labels):
            self.samples = samples
            self.labels = labels
        
        def __len__(self):
            return len(self.samples)
        
        def __getitem__(self, idx):
            return self.samples[idx], self.labels[idx]
    
    test_dataset = SimpleDataset(test_samples, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Run detection
    print(f"\nRunning STRIP detection on {len(test_dataset)} samples...")
    suspicious_indices = detector.detect(test_loader)
    
    print(f"  Detection complete")
    print(f"  Flagged as suspicious: {len(suspicious_indices)}/{len(test_dataset)} samples")
    
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
    csv_path = RESULTS_DIR / "strip.csv"

    results = {
        'exp_num': exp_num,
        'defense_name': 'STRIP',
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
    
    # Save to CSV
    save_results_to_csv(results, csv_path)


def main():
    parser = argparse.ArgumentParser(
        description="STRIP Defense for Backdoor Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python defenses/strip.py --attack badnet --model resnet18 --dataset cifar10 --poison_rate 0.05 --exp_num 1
  python defenses/strip.py --attack wanet --model resnet18 --dataset cifar10 --poison_rate 0.02 --exp_num 2
  python defenses/strip.py --attack bpp --model resnet18 --dataset cifar100 --poison_rate 0.003 --exp_num 3
        """
    )
    
    # Required arguments
    parser.add_argument('--attack', type=str, required=True,
                       help='Attack name (e.g., badnet, wanet, bpp)')
    parser.add_argument('--model', type=str, default='resnet18',
                       help='Model architecture (default: resnet18)')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name (cifar10, cifar100, imagenette)')
    parser.add_argument('--poison_rate', type=float, required=True,
                       help='Poison rate (e.g., 0.05, 0.02)')
    
    # STRIP parameters
    parser.add_argument('--strip_alpha', type=float, default=1.0,
                       help='STRIP blending coefficient (default: 1.0)')
    parser.add_argument('--N', type=int, default=100,
                       help='Number of clean samples to mix (default: 100)')
    parser.add_argument('--defense_fpr', type=float, default=0.1,
                       help='Desired false positive rate (default: 0.1)')
    
    # Other parameters
    parser.add_argument('--n_clean_calibration', type=int, default=500,
                       help='Number of clean samples for calibration (default: 500)')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size (default: 128)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')

    # New required argument for exp_num
    parser.add_argument('--exp_num', type=int, default=0,
                        help='Experiment number (default: 0)')

    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Run defense
    try:
        run_strip_defense(
            attack_name=args.attack,
            model_arch=args.model,
            dataset_name=args.dataset,
            poison_rate=args.poison_rate,
            strip_alpha=args.strip_alpha,
            N=args.N,
            defense_fpr=args.defense_fpr,
            n_clean_calibration=args.n_clean_calibration,
            batch_size=args.batch_size,
            exp_num=args.exp_num
        )
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
