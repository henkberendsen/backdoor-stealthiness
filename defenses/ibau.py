#!/usr/bin/env python3
"""
I-BAU: Adversarial Unlearning of Backdoors via Implicit Hypergradient

Implementation of I-BAU defense for backdoor mitigation.
This defense uses a minimax formulation to remove backdoors from a poisoned model
using only a small set of clean data.

Reference:
    Zeng et al. "Adversarial Unlearning of Backdoors via Implicit Hypergradient"
    ICLR 2022

Usage:
    python defenses/ibau.py --attack badnet --model resnet18 --dataset cifar10 --poison_rate 0.05 --exp_num 1
"""

import argparse
import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import csv
import copy
import time
from torch.utils.data import DataLoader, Subset
from torch.autograd import grad as torch_grad
from itertools import repeat
from typing import List, Callable
from torch import Tensor
from tqdm import tqdm
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


# =============================================================================
# I-BAU Hypergradient Implementation
# Based on https://github.com/YiZeng623/I-BAU
# =============================================================================

class DifferentiableOptimizer:
    """Differentiable optimizer for computing implicit hypergradients."""
    
    def __init__(self, loss_f, dim_mult, data_or_iter=None):
        """
        Args:
            loss_f: callable with signature (params, hparams, [data optional]) -> loss tensor
            data_or_iter: (x, y) or iterator over the data needed for loss_f
        """
        self.data_iterator = None
        if data_or_iter:
            self.data_iterator = data_or_iter if hasattr(data_or_iter, '__next__') else repeat(data_or_iter)

        self.loss_f = loss_f
        self.dim_mult = dim_mult
        self.curr_loss = None

    def get_opt_params(self, params):
        opt_params = [p for p in params]
        opt_params.extend([torch.zeros_like(p) for p in params for _ in range(self.dim_mult-1)])
        return opt_params

    def step(self, params, hparams, create_graph):
        raise NotImplementedError

    def __call__(self, params, hparams, create_graph=True):
        with torch.enable_grad():
            return self.step(params, hparams, create_graph)

    def get_loss(self, params, hparams):
        if self.data_iterator:
            data = next(self.data_iterator)
            self.curr_loss = self.loss_f(params, hparams, data)
        else:
            self.curr_loss = self.loss_f(params, hparams)
        return self.curr_loss


class GradientDescent(DifferentiableOptimizer):
    """Gradient descent optimizer for inner optimization."""
    
    def __init__(self, loss_f, step_size, data_or_iter=None):
        super(GradientDescent, self).__init__(loss_f, dim_mult=1, data_or_iter=data_or_iter)
        self.step_size_f = step_size if callable(step_size) else lambda x: step_size

    def step(self, params, hparams, create_graph):
        loss = self.get_loss(params, hparams)
        sz = self.step_size_f(hparams)
        return gd_step(params, loss, sz, create_graph=create_graph)


def gd_step(params, loss, step_size, create_graph=True):
    """Perform a gradient descent step."""
    grads = torch.autograd.grad(loss, params, create_graph=create_graph)
    return [w - step_size * g for w, g in zip(params, grads)]


def grad_unused_zero(output, inputs, grad_outputs=None, retain_graph=False, create_graph=False):
    """Compute gradients, returning zeros for unused inputs."""
    grads = torch.autograd.grad(output, inputs, grad_outputs=grad_outputs, allow_unused=True,
                                retain_graph=retain_graph, create_graph=create_graph)

    def grad_or_zeros(grad, var):
        return torch.zeros_like(var) if grad is None else grad

    return tuple(grad_or_zeros(g, v) for g, v in zip(grads, inputs))


def get_outer_gradients(outer_loss, params, hparams, retain_graph=True):
    """Get gradients with respect to params and hyperparams."""
    grad_outer_w = grad_unused_zero(outer_loss, params, retain_graph=retain_graph)
    grad_outer_hparams = grad_unused_zero(outer_loss, hparams, retain_graph=retain_graph)
    return grad_outer_w, grad_outer_hparams


def update_tensor_grads(hparams, grads):
    """Update tensor gradients."""
    for l, g in zip(hparams, grads):
        if l.grad is None:
            l.grad = torch.zeros_like(l)
        if g is not None:
            l.grad += g


def cat_list_to_tensor(list_tx):
    """Concatenate list of tensors into a single tensor."""
    return torch.cat([xx.reshape([-1]) for xx in list_tx])


def fixed_point(params: List[Tensor],
                hparams: List[Tensor],
                K: int,
                fp_map: Callable[[List[Tensor], List[Tensor]], List[Tensor]],
                outer_loss: Callable[[List[Tensor], List[Tensor]], Tensor],
                tol=1e-10,
                set_grad=True,
                stochastic=False) -> List[Tensor]:
    """
    Compute hypergradient using fixed point method.
    
    Args:
        params: Output of inner solver procedure
        hparams: Outer variables (hyperparameters), each needs requires_grad=True
        K: Maximum number of fixed point iterations
        fp_map: Fixed point map defining the inner problem
        outer_loss: Computes outer objective
        tol: Tolerance for early stopping
        set_grad: If True, set t.grad to hypergradient for every t in hparams
        stochastic: Set True when fp_map is not deterministic
        
    Returns:
        List of hypergradients for each element in hparams
    """
    params = [w.detach().requires_grad_(True) for w in params]
    o_loss = outer_loss(params, hparams)
    grad_outer_w, grad_outer_hparams = get_outer_gradients(o_loss, params, hparams)

    if not stochastic:
        w_mapped = fp_map(params, hparams)

    vs = [torch.zeros_like(w) for w in params]
    vs_vec = cat_list_to_tensor(vs)
    
    for k in range(K):
        vs_prev_vec = vs_vec

        if stochastic:
            w_mapped = fp_map(params, hparams)
            vs = torch_grad(w_mapped, params, grad_outputs=vs, retain_graph=False)
        else:
            vs = torch_grad(w_mapped, params, grad_outputs=vs, retain_graph=True)

        vs = [v + gow for v, gow in zip(vs, grad_outer_w)]
        vs_vec = cat_list_to_tensor(vs)
        if float(torch.norm(vs_vec - vs_prev_vec)) < tol:
            break

    if stochastic:
        w_mapped = fp_map(params, hparams)

    grads = torch_grad(w_mapped, hparams, grad_outputs=vs, allow_unused=True)
    grads = [g + v if g is not None else v for g, v in zip(grads, grad_outer_hparams)]

    if set_grad:
        update_tensor_grads(hparams, grads)

    return grads


# =============================================================================
# I-BAU Defense Implementation
# =============================================================================

def run_ibau_defense(model, clean_dataset, device, n_rounds=5, K=5, 
                     outer_lr=0.002, inner_lr=0.1, batch_lr=0.05,
                     portion=0.02, batch_size=128):
    """
    Run I-BAU defense to unlearn backdoor from model.
    
    Args:
        model: Poisoned model to defend
        clean_dataset: Clean dataset for unlearning
        device: torch device
        n_rounds: Number of outer rounds
        K: Number of fixed point iterations
        outer_lr: Learning rate for outer optimizer
        inner_lr: Learning rate for inner optimizer
        batch_lr: Learning rate for batch perturbation
        portion: Portion of samples to apply perturbation
        batch_size: Batch size
        
    Returns:
        Defended model
    """
    val_dataloader = DataLoader(clean_dataset, batch_size=batch_size, 
                                shuffle=True, drop_last=True)
    
    # Collect batches
    images_list, labels_list = [], []
    for images, labels in val_dataloader:
        images_list.append(images)
        labels_list.append(labels)
    
    if len(images_list) == 0:
        raise ValueError("No data batches available for I-BAU defense")
    
    def loss_inner(perturb, model_params):
        """Inner loss for finding adversarial perturbation."""
        images = images_list[0].to(device)
        labels = labels_list[0].long().to(device)
        per_img = images + perturb[0]
        per_logits = model.forward(per_img)
        loss = F.cross_entropy(per_logits, labels, reduction='none')
        loss_regu = torch.mean(-loss) + 0.001 * torch.pow(torch.norm(perturb[0]), 2)
        return loss_regu
    
    # Variable to track current batch number
    batchnum_container = [0]
    
    def loss_outer(perturb, model_params):
        """Outer loss for model unlearning."""
        batchnum = batchnum_container[0]
        images = images_list[batchnum].to(device)
        labels = labels_list[batchnum].long().to(device)
        patching = torch.zeros_like(images, device=device)
        number = images.shape[0]
        rand_idx = random.sample(list(np.arange(number)), int(number * portion))
        patching[rand_idx] = perturb[0]
        unlearn_imgs = images + patching
        logits = model(unlearn_imgs)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, labels)
        return loss
    
    model = model.to(device)
    outer_opt = torch.optim.Adam(model.parameters(), lr=outer_lr)
    inner_opt = GradientDescent(loss_inner, inner_lr)
    
    model.train()
    
    # Initialize batch perturbation
    sample_img = clean_dataset[0][0]
    if isinstance(sample_img, torch.Tensor):
        batch_pert = torch.zeros_like(sample_img.unsqueeze(0), 
                                       requires_grad=True, device=device)
    else:
        # Convert PIL to tensor
        to_tensor = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])
        sample_tensor = to_tensor(sample_img)
        batch_pert = torch.zeros_like(sample_tensor.unsqueeze(0), 
                                       requires_grad=True, device=device)
    
    batch_opt = torch.optim.Adam(params=[batch_pert], lr=batch_lr)
    
    print(f"  Starting I-BAU unlearning ({n_rounds} rounds)...")
    
    for round_idx in range(n_rounds):
        # Step 1: Find adversarial perturbation
        for images, labels in val_dataloader:
            images = images.to(device)
            ori_lab = torch.argmax(model.forward(images), axis=1).long()
            per_logits = model.forward(images + batch_pert)
            loss = -F.cross_entropy(per_logits, ori_lab) + 0.001 * torch.pow(torch.norm(batch_pert), 2)
            batch_opt.zero_grad()
            loss.backward(retain_graph=True)
            batch_opt.step()
        
        # Normalize perturbation
        pert = batch_pert * min(1, 10 / torch.norm(batch_pert))
        
        # Step 2: Unlearn step
        for batchnum in range(len(images_list)):
            batchnum_container[0] = batchnum
            outer_opt.zero_grad()
            fixed_point(pert, list(model.parameters()), K, inner_opt, loss_outer)
            outer_opt.step()
        
        print(f"    Round {round_idx + 1}/{n_rounds} complete")
    
    model.eval()
    return model


def evaluate_model(model, dataloader, device):
    """
    Evaluate model accuracy on a dataloader.
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader for evaluation
        device: torch device
        
    Returns:
        dict with 'accuracy' and 'loss'
    """
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.long().to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total if total > 0 else 0
    avg_loss = total_loss / total if total > 0 else 0
    
    return {'accuracy': accuracy, 'loss': avg_loss}


def verify_attack_exists(attack_name, model_arch, dataset_name, poison_rate):
    """Verify that the attack directory exists."""
    exp_id = experiment_variable_identifier(model_arch, dataset_name, poison_rate)
    attack_dir = RECORD_DIR / f"{attack_name}_{exp_id}"
    
    if not attack_dir.exists():
        # Try with formatted poison rate
        pr_str = str(poison_rate).replace('.', '-')
        exp_id_alt = f"{model_arch}_{dataset_name}_p{pr_str}"
        attack_dir_alt = RECORD_DIR / f"{attack_name}_{exp_id_alt}"
        
        if attack_dir_alt.exists():
            return attack_dir_alt
        
        raise FileNotFoundError(
            f"Attack directory not found: {attack_dir}\n"
            f"Also tried: {attack_dir_alt}"
        )
    
    return attack_dir


def load_attack_data(attack_name, dataset_name, model_arch, poison_rate, device):
    """
    Load attack data using eval_utils dispatcher.
    
    Returns:
        dict with 'model', 'bd_test', 'clean_test', 'clean_train', 'attack_record', 'clean_record'
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
    
    # Get test datasets
    bd_test = bd_record.get('test', None)
    if bd_test is None:
        raise ValueError(f"No test dataset found for attack {attack_name}")
    
    # Get training data for unlearning (try 'train' first, then 'train_transformed')
    clean_train = clean_record.get('train', None)
    if clean_train is None:
        clean_train = clean_record.get('train_transformed', None)
    
    # Use clean test from clean_record
    clean_test = clean_record['test']
    
    print(f"  Loaded successfully")
    print(f"    - Model: {model_arch}")
    print(f"    - Backdoor test samples: {len(bd_test)}")
    print(f"    - Clean test samples: {len(clean_test)}")
    if clean_train:
        print(f"    - Clean train samples: {len(clean_train)}")
    
    return {
        'model': model,
        'bd_test': bd_test,
        'clean_test': clean_test,
        'clean_train': clean_train,
        'attack_record': bd_record,
        'clean_record': clean_record
    }


def save_results_to_csv(results, csv_path):
    """Save results to CSV file."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if file exists to determine if we need headers
    file_exists = csv_path.exists()
    
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'exp_num', 'defense_name', 'attack', 'model', 'dataset', 
            'poison_rate', 'CDA_before', 'ASR_before', 'CDA_after', 'ASR_after',
            'elapsed_time'
        ])
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(results)
    
    print(f"\nResults saved to: {csv_path}")


def run_ibau(attack_name, model_arch, dataset_name, poison_rate,
             n_rounds=5, K=5, outer_lr=0.002, inner_lr=0.1, batch_lr=0.05,
             n_clean_samples=5000, batch_size=128, exp_num=None):
    """
    Run I-BAU defense on a backdoor attack.
    
    Args:
        attack_name: Name of the attack
        model_arch: Model architecture
        dataset_name: Dataset name
        poison_rate: Poison rate
        n_rounds: Number of I-BAU rounds
        K: Number of fixed point iterations
        outer_lr: Learning rate for outer optimizer
        inner_lr: Learning rate for inner optimizer
        batch_lr: Learning rate for batch perturbation
        n_clean_samples: Number of clean samples for unlearning
        batch_size: Batch size
        exp_num: Experiment number
    """
    print("=" * 70)
    print("I-BAU: Adversarial Unlearning of Backdoors")
    print("=" * 70)
    print(f"Attack: {attack_name}")
    print(f"Model: {model_arch}")
    print(f"Dataset: {dataset_name}")
    print(f"Poison Rate: {poison_rate}")
    print(f"I-BAU Parameters: rounds={n_rounds}, K={K}, outer_lr={outer_lr}")
    print("=" * 70)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    # Verify attack exists
    print(f"\nVerifying attack directory...")
    attack_dir = verify_attack_exists(attack_name, model_arch, dataset_name, poison_rate)
    print(f"  Found: {attack_dir}")
    
    # Load attack data
    print(f"\nLoading attack data...")
    attack_data = load_attack_data(attack_name, dataset_name, model_arch, poison_rate, device)
    model = attack_data['model']
    bd_test = attack_data['bd_test']
    clean_test = attack_data['clean_test']
    clean_train = attack_data['clean_train']
    
    # Create transform for tensor conversion
    to_tensor_transform = T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True)
    ])
    
    # Wrap datasets with transforms
    bd_test_transformed = TransformedDataset(bd_test, transform=to_tensor_transform)
    clean_test_transformed = TransformedDataset(clean_test, transform=to_tensor_transform)
    
    # Create dataloaders for evaluation
    bd_test_loader = DataLoader(bd_test_transformed, batch_size=batch_size, shuffle=False)
    clean_test_loader = DataLoader(clean_test_transformed, batch_size=batch_size, shuffle=False)
    
    # Evaluate BEFORE defense
    print(f"\nEvaluating model BEFORE defense...")
    cda_before = evaluate_model(model, clean_test_loader, device)
    asr_before = evaluate_model(model, bd_test_loader, device)
    print(f"  Clean Data Accuracy (CDA): {cda_before['accuracy']:.4f}")
    print(f"  Attack Success Rate (ASR): {asr_before['accuracy']:.4f}")
    
    # Prepare clean data for unlearning
    print(f"\nPreparing clean data for unlearning...")
    if clean_train is not None:
        clean_train_transformed = TransformedDataset(clean_train, transform=to_tensor_transform)
        # Sample subset for unlearning
        n_samples = min(n_clean_samples, len(clean_train_transformed))
        subset_indices = np.random.choice(len(clean_train_transformed), n_samples, replace=False)
        unlearn_dataset = Subset(clean_train_transformed, subset_indices)
    else:
        # Fall back to clean test if no training data
        n_samples = min(n_clean_samples, len(clean_test_transformed))
        subset_indices = np.random.choice(len(clean_test_transformed), n_samples, replace=False)
        unlearn_dataset = Subset(clean_test_transformed, subset_indices)
    
    print(f"  Using {len(unlearn_dataset)} clean samples for unlearning")
    
    # Run I-BAU defense
    print(f"\nRunning I-BAU defense...")
    start_time = time.perf_counter()
    
    # Make a copy of the model for defense
    model_defended = copy.deepcopy(model)
    model_defended = run_ibau_defense(
        model_defended, 
        unlearn_dataset, 
        device,
        n_rounds=n_rounds,
        K=K,
        outer_lr=outer_lr,
        inner_lr=inner_lr,
        batch_lr=batch_lr,
        batch_size=batch_size
    )
    
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"  I-BAU defense complete in {elapsed_time:.2f} seconds")
    
    # Evaluate AFTER defense
    print(f"\nEvaluating model AFTER defense...")
    cda_after = evaluate_model(model_defended, clean_test_loader, device)
    asr_after = evaluate_model(model_defended, bd_test_loader, device)
    print(f"  Clean Data Accuracy (CDA): {cda_after['accuracy']:.4f}")
    print(f"  Attack Success Rate (ASR): {asr_after['accuracy']:.4f}")
    
    # Print results
    print(f"\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"{'Metric':<30} {'Before':<15} {'After':<15} {'Change':<15}")
    print("-" * 70)
    print(f"{'Clean Data Accuracy (CDA)':<30} {cda_before['accuracy']:<15.4f} {cda_after['accuracy']:<15.4f} {cda_after['accuracy'] - cda_before['accuracy']:+.4f}")
    print(f"{'Attack Success Rate (ASR)':<30} {asr_before['accuracy']:<15.4f} {asr_after['accuracy']:<15.4f} {asr_after['accuracy'] - asr_before['accuracy']:+.4f}")
    print("-" * 70)
    print(f"Defense Time: {elapsed_time:.2f} seconds")
    print("=" * 70)
    
    # Prepare results for CSV
    csv_path = RESULTS_DIR / "ibau.csv"
    
    results = {
        'exp_num': exp_num,
        'defense_name': 'I-BAU',
        'attack': attack_name,
        'model': model_arch,
        'dataset': dataset_name,
        'poison_rate': poison_rate,
        'CDA_before': round(cda_before['accuracy'], 4),
        'ASR_before': round(asr_before['accuracy'], 4),
        'CDA_after': round(cda_after['accuracy'], 4),
        'ASR_after': round(asr_after['accuracy'], 4),
        'elapsed_time': round(elapsed_time, 2)
    }
    
    # Save to CSV
    save_results_to_csv(results, csv_path)


def main():
    parser = argparse.ArgumentParser(
        description="I-BAU Defense for Backdoor Mitigation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python defenses/ibau.py --attack badnet --model resnet18 --dataset cifar10 --poison_rate 0.05 --exp_num 1
  python defenses/ibau.py --attack wanet --model resnet18 --dataset cifar10 --poison_rate 0.02 --exp_num 2
  python defenses/ibau.py --attack bpp --model resnet18 --dataset cifar100 --poison_rate 0.003 --exp_num 3
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
    
    # I-BAU parameters
    parser.add_argument('--n_rounds', type=int, default=5,
                       help='Number of I-BAU rounds (default: 5)')
    parser.add_argument('--K', type=int, default=5,
                       help='Number of fixed point iterations (default: 5)')
    parser.add_argument('--outer_lr', type=float, default=0.002,
                       help='Outer optimizer learning rate (default: 0.002)')
    parser.add_argument('--inner_lr', type=float, default=0.1,
                       help='Inner optimizer learning rate (default: 0.1)')
    parser.add_argument('--batch_lr', type=float, default=0.05,
                       help='Batch perturbation learning rate (default: 0.05)')
    
    # Other parameters
    parser.add_argument('--n_clean_samples', type=int, default=5000,
                       help='Number of clean samples for unlearning (default: 5000)')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size (default: 128)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
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
        run_ibau(
            attack_name=args.attack,
            model_arch=args.model,
            dataset_name=args.dataset,
            poison_rate=args.poison_rate,
            n_rounds=args.n_rounds,
            K=args.K,
            outer_lr=args.outer_lr,
            inner_lr=args.inner_lr,
            batch_lr=args.batch_lr,
            n_clean_samples=args.n_clean_samples,
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
