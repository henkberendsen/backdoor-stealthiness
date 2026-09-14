import copy
import numpy as np
import os
import pandas as pd
import sys
import torch
import torchvision
import torchvision.transforms.v2 as T
from torchvision import transforms
from torch.utils.data import DataLoader

from itertools import product
from matplotlib import pyplot as plt
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(DEVICE)

if torch.cuda.is_available(): 
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DATASET = "cifar10"
DATASETS = [DATASET]

# Records, datasets and intermediate results all live under one directory: large_files/ next to
# this file by default, or the directory named by the BACKDOOR_STEALTHINESS_DATA variable.
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
LARGE_FILES = os.environ.get("BACKDOOR_STEALTHINESS_DATA", os.path.join(REPO_ROOT, "large_files"))
DATA_DIR = os.path.join(LARGE_FILES, "data")
RECORD_DIR = os.path.join(LARGE_FILES, "record")
RESULT_DIR = LARGE_FILES
TARGET_CLASS = 0
MODEL_ARCH = 'resnet18'
MODEL_ARCHITECTURES = [MODEL_ARCH]

NORMALIZATION_DICT = {
    "cifar10": ([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]),
    "cifar100": ([0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2762]),
    "tiny": ([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
    "imagenette": ([0.4671, 0.4593, 0.4306], [0.2692, 0.2657, 0.2884])
}

transform_train = {
    "cifar10": transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(*NORMALIZATION_DICT["cifar10"])
    ]),
    "cifar100": transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(*NORMALIZATION_DICT["cifar100"])
    ]),  
    "tiny": transforms.Compose([
        transforms.RandomCrop(64, padding=4),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(*NORMALIZATION_DICT["tiny"])
    ]),
    "imagenette": transforms.Compose([
        transforms.RandomCrop(80, padding=4),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(*NORMALIZATION_DICT["imagenette"])
    ])  
}

transform_test = {
    "cifar10": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*NORMALIZATION_DICT["cifar10"])
    ]),
    "cifar100": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*NORMALIZATION_DICT["cifar100"])
    ]),  
    "tiny": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*NORMALIZATION_DICT["tiny"])
    ]),
    "imagenette": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*NORMALIZATION_DICT["imagenette"])
    ])  
}


TRANSFORM_DICT = {}

TRANSFORM_DICT['cifar10_train'] = transform_train['cifar10']
TRANSFORM_DICT['cifar10_test'] = transform_test['cifar10']

TRANSFORM_DICT['tiny_train'] = transform_train['tiny']
TRANSFORM_DICT['tiny_test'] = transform_test['tiny']




BATCH_SIZE = 20 if DATASET == "imagenette" else 100

POISON_RATES = [
    lambda atk: 0.007 if DATASET == "cifar100" and atk in ["narcissus", "grond"] else 0.05,
    lambda _: 0.003
]

# Highest poison rate when comparing or visualizing trainset stealthiness 
DEFAULT_POISON_RATE = POISON_RATES[0]

# Below constants allow multiple datasets and model architectures to be compared at the same time
# We use this for one input-space stealthiness experiment

N_CLASSES_DICT = {
    "cifar10": 10,
    "cifar100": 100,
    "tiny": 200,
    "imagenette": 10
}

IMG_SIZE_DICT = {
    "cifar10": (32, 32),
    "cifar100": (32, 32),
    "tiny": (64, 64),
    "imagenette": (80, 80)
}

# Which index to use when comparing poisoned images, for each dataset
IMG_INDEX_DICT = {
    "cifar10": {
        "train": [3, 50, 72, -4],
        "test": 4
    },
    "cifar100": {
        "train": [1, 11, 15, 21],
        "test": 206
    },
    "tiny": {
        "train": [2, 3, 12, 4],
        "test": 17
    },
    "imagenette": {
        "train": [2, 3, 12, 4],
        "test": 17
    }
}

ATK_PPRINT_DICT = {
    "badnet": "BadNets",
    "blended": "Blend",
    "wanet": "WaNet",
    "bpp": "BppAttack",
    "adaptive_patch": "Adap-Patch",
    "adaptive_blend": "Adap-Blend",
    "dfst": "DFST",
    "dfba": "DFBA",
    "narcissus": "Narcissus",
    "grond": "Grond"
}



class BackdoorDataset(Dataset):
    def __init__(self, bd_dataset, classes, target_class, original_labels, poison_lookup, cross_lookup=None):
        self.bd_dataset = bd_dataset
        self.classes = classes
        self.target_class = target_class
        self.original_labels = original_labels
        self.poison_lookup = poison_lookup
        self.cross_lookup = cross_lookup

        # Create the poisoned labels by copying the original labels, and setting the poisoned indices to the target class
        self.poisoned_labels = self.original_labels.copy()
        self.poisoned_labels[self.poison_lookup] = self.target_class

    def __len__(self):
        return len(self.bd_dataset)
    
    def __getitem__(self, index):
        return self.bd_dataset.__getitem__(index) 
    
# Wrapper around BackdoorBench's utils.bd_dataset_v2.dataset_wrapper_with_transform
class BackdoorBenchDataset(BackdoorDataset):
    def __init__(self, bb_dataset, target_class, replace_transform=None):
        classes = bb_dataset.wrapped_dataset.dataset.classes
        original_labels = np.array(bb_dataset.wrapped_dataset.dataset.targets)
        poison_lookup = bb_dataset.wrapped_dataset.poison_indicator == 1
        cross_lookup = bb_dataset.wrapped_dataset.poison_indicator == 2

        if replace_transform:
            bb_dataset.wrap_img_transform = replace_transform

        super().__init__(bb_dataset, classes, target_class, original_labels, poison_lookup, cross_lookup)

    # BackdoorBench dataset returns more than just the image and label
    def __getitem__(self, index):
        img, bd_label, dataset_idx, poison_bit, clean_label = self.bd_dataset.__getitem__(index) 
                       
        return img, bd_label

# Convert Adap-Blend/Patch folder of poisoned images to BackdoorDataset child class  
class AdapDataset(BackdoorDataset):
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
            poison_indices = range(n_samples)
            cover_indices = []

        # Replace benign images by their poisoned/cross versions for the indices specified above
        for i in np.concatenate([poison_indices, cover_indices]):
            i = int(i)
            img = Image.open(os.path.join(bd_dataset_path, f"{i}.png"))
            bd_dataset.data[i] = np.asarray(img)

            # Also set label to target class for poisoned images
            if i in poison_indices:
                bd_dataset.targets[i] = target_class
        
        self.bd_dataset = bd_dataset
        self.transform = bd_dataset.transform

        poison_lookup = np.array([i in poison_indices for i in range(n_samples)])
        cross_lookup = np.array([i in cover_indices for i in range(n_samples)])

        super().__init__(bd_dataset, classes, target_class, original_labels, poison_lookup, cross_lookup)

    def __getitem__(self, index):
        target = self.bd_dataset.targets[index]
        img = self.bd_dataset.data[index]

        if self.transform is not None:
            img = Image.fromarray(img)
            img = self.transform(img)
        return img, target

# Convert DFST poison_data.pt file to BackdoorDataset child class
class DFSTDataset(BackdoorDataset):
    def __init__(self, bd_record_path, target_class, split, clean_dataset):
        classes = clean_dataset.classes
        original_labels = np.array(clean_dataset.targets)
        n_samples = len(clean_dataset)
        bd_dataset = copy.deepcopy(clean_dataset)
        self.transform = bd_dataset.transform
        bd_dataset_path = os.path.join(bd_record_path, "poison_data.pt")

        # DFST saves poisoned versions of each image that is not of the target class
        poison_data = torch.load(bd_dataset_path, weights_only=False)
        non_target_poisoned = poison_data[split]

        if split == "train":
            # DFST randomly chooses poisoned samples each batch and epoch and replaces them by poisoned images that are not of the target class
            # These poisoned images are saved to a file
            # We have modified the file to only include a subset of these images according to the poison rate, and have also saved their indices
            poison_indices = poison_data["train_indices"]

            # Mapping from indices in the trainset without the target class to indices in the full trainset
            true_indices = np.arange(n_samples)[original_labels != TARGET_CLASS].squeeze()
        else:
            # The testset is fully poisoned
            poison_indices = range(n_samples)

            # The clean testset also filters the target class, so we do not have to convert the indices
            true_indices = poison_indices 

        # Initialize poison_lookup and cross_lookup
        poison_lookup = np.full(n_samples, False) 
        cross_lookup = np.full(n_samples, False) 

        # Replace benign images by poisoned ones for the poisoned indices in the DFST file, and their corresponding true indices in the dataset
        for i, poison_idx in enumerate(poison_indices):
            true_idx = true_indices[poison_idx]
            poison_lookup[true_idx] = True

            # Get poisoned image, put it in the backdoor dataset and change its label to the target class
            poisoned_img = non_target_poisoned[i]
            bd_dataset.data[true_idx] = (poisoned_img * 255).permute(1, 2, 0).numpy()
            bd_dataset.targets[true_idx] = target_class 

        super().__init__(bd_dataset, classes, target_class, original_labels, poison_lookup, cross_lookup)

    def __getitem__(self, index):
        target = self.bd_dataset.targets[index]
        img = self.bd_dataset.data[index]

        if self.transform is not None:
            img = Image.fromarray(img)
            img = self.transform(img)
        return img, target


# Apply DFBA trigger to benign testset
class DFBADataset(BackdoorDataset):
    def __init__(self, clean_testset, target_class, delta, mask):
        # Get arguments expected by BackdoorDataset
        classes = clean_testset.classes
        original_labels = np.array(clean_testset.targets) # Clean-label attack, so labels do not change
        n_samples = len(clean_testset)
        poison_lookup = np.full(n_samples, True)
        cross_lookup = np.full(n_samples, False) 
        bd_testset = copy.deepcopy(clean_testset)

        # Poison all test images
        for i in range(n_samples):
            # Make color channels first dimension instead of last in order to apply mask to the image
            img = bd_testset.data[i].transpose(2, 0, 1)

            # Combine image and trigger using mask
            bd_img = img * (1 - mask) + (delta * 255) * mask

            # Reverse reordering of dimensions
            bd_testset.data[i] = bd_img.transpose(1, 2, 0)

            # Set label of image to target class
            bd_testset.targets[i] = target_class

        self.transform = bd_testset.transform
        self.bd_testset = bd_testset

        super().__init__(bd_testset, classes, target_class, original_labels, poison_lookup, cross_lookup)

    def __getitem__(self, index):
        target = self.bd_testset.targets[index]
        img = self.bd_testset.data[index]

        if self.transform is not None:
            img = self.transform(img)
        return img, target


# Add absolute path to ./grond to the system PATH to prevent error in class below
grond_dir = os.path.abspath("./grond")
sys.path.append(grond_dir)
from grond.poison_loader import POI, POI_TEST

# Wrapper around Grond's POI, POI_TEST (abstraction of CIFAR10_POI, CIFAR10_POI_TEST for any dataset)
class GrondDataset(BackdoorDataset):
    def __init__(self, dataset, transform, target_class, record_path, train=True):
        # Reconstruct poisoned dataset used in attack
        if train:
            poison_indices = torch.load(os.path.join(record_path, "poison_indices.pth"))   
            poi_dataset = POI(dataset, root=DATA_DIR, 
                              poison_rate=None, # poison rate is unused as we pass poison_indices directly
                              transform=transform, poison_indices=poison_indices,
                              target_cls=target_class, upgd_path=record_path)
        else:
            poi_dataset = POI_TEST(dataset, root=DATA_DIR,
                                   transform=transform, exclude_target=True,
                                   target_cls=target_class, upgd_path=record_path)

        # Get arguments expected by BackdoorDataset
        classes = poi_dataset.cleanset.classes
        original_labels = np.array(poi_dataset.targets) # Clean-label attack, so labels do not change
        n_samples = len(poi_dataset)
        poison_lookup = np.array([i in poison_indices for i in range(n_samples)]) if train else np.full(n_samples, True)
        cross_lookup = np.full(n_samples, False) 

        super().__init__(poi_dataset, classes, target_class, original_labels, poison_lookup, cross_lookup)



# ----------------- Input-space
"""Implementation of all input-space stealthiness metrics used in the study.

Each metric requires the parameters img1 and img2, which are tensors of shape Nx3xHxW, representing batches of N RGB images of size HxW.
The metrics calculate the mean similarity over the two batches, and they can also be used for single images by passing tensors with N=1.
"""

from numpy.linalg import norm
from sklearn.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import lpips
import torch
from torch.nn.functional import softmax
from scipy.special import rel_entr
from imagehash import phash
import torchvision.transforms.v2 as T
from torchmetrics.functional.image import spectral_angle_mapper

# Load neural networks globally for faster metric calculation
loss_fn = lpips.LPIPS(net="alex", verbose=False).to(DEVICE, non_blocking=True)
inception_v3 = torch.hub.load("pytorch/vision:v0.10.0", "inception_v3", pretrained=True).to(DEVICE, non_blocking=True)
inception_v3.eval()

def lp_norm(p, imgs):
    return np.mean([norm(img.flatten(), ord=p) for img in imgs])

def l1_distance(img1, img2):
    return lp_norm(1, img1 - img2)

def l2_distance(img1, img2):
    return lp_norm(2, img1 - img2)

def linf_distance(img1, img2):
    print('img1: ', img1[0])
    print('img2: ', img2[0])
    return lp_norm(np.inf, img1 - img2)

def MSE(img1, img2):
    return mean_squared_error(img1.flatten(), img2.flatten())

def PSNR(img1, img2):
    with np.errstate(divide="ignore"): # Ignore divide by zero warnings which occur if img1 == img2
        # data_range1 = img1.max() - img1.min()
        # data_range2 = img2.max() - img2.min()
        return peak_signal_noise_ratio(img1.numpy(), img2.numpy())

def SSIM(img1, img2):
    SSIM_per_image = lambda x, y: structural_similarity(x, y, data_range=1, channel_axis=0)
    SSIM_values = np.array(list(map(SSIM_per_image, img1.numpy(), img2.numpy())))

    return SSIM_values.mean()

def pHash(img1, img2):
    def pHash_per_image(x, y):
        tensor_to_pil = T.ToPILImage()
        pil1 = tensor_to_pil(x)
        pil2 = tensor_to_pil(y)

        hash1 = phash(pil1)
        hash2 = phash(pil2)
    
        def hamming_distance(hash1, hash2):
            def phash_to_bool_array(phash):
                return phash.hash.flatten() # Convert 8x8 boolean representation of 64-bit hash to 1D array of length 64
                
            bool_array1 = phash_to_bool_array(hash1)
            bool_array2 = phash_to_bool_array(hash2)

            return np.where(bool_array1 != bool_array2)[0].size
        
        return (1 - (hamming_distance(hash1, hash2) / 64.0)) * 100
    
    pHash_values = np.array(list(map(pHash_per_image, img1, img2)))
    
    return pHash_values.mean()

def LPIPS(img1, img2):
    with torch.no_grad():
        normalize = T.Normalize([0.5], [0.5]) # Normalization required by LPIPS
        img1, img2 = img1.to(DEVICE, non_blocking=True), img2.to(DEVICE, non_blocking=True)
        
        return loss_fn.forward(normalize(img1), normalize(img2)).mean().item()

def IS(img1, img2):
    with torch.no_grad():
        preprocess = T.Compose([
            T.Resize(299), # Model expects 299x299 images
            T.CenterCrop(299),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), # Normalization required by Inception v3 model
        ])
               
        out1 = inception_v3(preprocess(img1.to(DEVICE, non_blocking=True)))
        out2 = inception_v3(preprocess(img2.to(DEVICE, non_blocking=True)))

        # Apply softmax to turn Inception v3 output into probabilities
        preds1 = softmax(out1, dim=1)
        preds2 = softmax(out2, dim=1)
        
        return rel_entr(preds1.cpu().numpy(), preds2.cpu().numpy()).sum(axis=1).mean() # Relative entropy = Kullback-Leibler divergence

def SAM(img1, img2):
    # Reduce range of images from [0, 1] to [1e-8, 1], as 0 vectors can cause NaN result
    img1_clamped = torch.clamp(img1, 1e-8, 1)
    img2_clamped = torch.clamp(img2, 1e-8, 1)

    return spectral_angle_mapper(img1_clamped, img2_clamped, reduction='elementwise_mean').item()




from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Fixed t-SNE configuration for the feature-space metrics (SS and CDBI): two components,
# perplexity 30, PCA initialisation and a fixed random state, so that the embedding -- and
# therefore both metrics -- are reproducible across runs.
TSNE_SEED = 0
TSNE_KWARGS = dict(n_components=2, perplexity=30.0, init="pca", learning_rate="auto",
                   random_state=TSNE_SEED)

def create_tsne(trainset, feature_path, skip_misclassified=False, show_plot=False, save_dst=None):
    # Load target-label predictions, features the train data indices they belong to from the specified file
    feature_dict = torch.load(feature_path, weights_only=False)
    predictions = feature_dict["predictions"]
    features = feature_dict["features"]
    target_label_indices = feature_dict["indices"]
    
    # Optionally filter features corresponding to wrongly classified inputs
    if skip_misclassified:
        correctly_classified = predictions == trainset.poisoned_labels
        features = features[correctly_classified]
        target_label_indices = target_label_indices[correctly_classified]

    # Perform dimensionality reduction on the extracted features
    def reduce_feature_dimensionality(features):
        return TSNE(**TSNE_KWARGS).fit_transform(features)

    features_embedded = reduce_feature_dimensionality(features)

    # Split the features with the target label into benign, poisoned and cross features, and return the poisoned boolean index array
    def split_features(features, poison_lookup, cross_lookup, indices):
        is_poisoned = poison_lookup[indices]
        is_cross = cross_lookup[indices]
        is_benign = np.logical_not(np.logical_or(is_poisoned, is_cross))

        benign = features[is_benign]
        poisoned = features[is_poisoned]
        cross = features[is_cross]
        
        return benign, poisoned, cross, is_poisoned

    features_benign, features_poisoned, features_cross, is_poisoned = split_features(features_embedded, trainset.poison_lookup, trainset.cross_lookup, target_label_indices)
    gt_labels = trainset.original_labels[target_label_indices][is_poisoned]

    # Visualize the latent separability of the model, styling from https://github.com/Unispac/Circumventing-Backdoor-Defenses/blob/master/visualize.py
    def plot(features_benign, features_poisoned, features_cross, gt_labels, highlight_cross=False, groupby_gt=False, show_plot=False, save_dst=None):
        def plot_benign(features):
            plt.scatter(features[:, 0], features[:, 1], label="Benign", marker='o', s=5, color="blue", alpha=1.0)

        # Plot cross samples separately if highlight_cross is true, otherwise consider them as benign samples
        if highlight_cross:
            plot_benign(features_benign)
            plt.scatter(features_cross[:, 0], features_cross[:, 1], label="Cross", marker='v', s=8, color='green', alpha=0.7)
        else:
            plot_benign(np.concatenate([features_benign, features_cross], axis=0))

        # Group the poisoned features by their corresponding ground truth label, to show how each class forms a separate cluster 
        if groupby_gt:
            class_colors = ['#ff001c', '#ff0055', '#ff008e', '#ff00c6', '#ff00ff', 
                            '#ff1c00', '#ff5500', '#ff8e00', '#ffc600', '#ffff00']

            for i, c in enumerate(trainset.classes):
                c_indices = gt_labels == i
                c_features = features_poisoned[c_indices]
                plt.scatter(c_features[:, 0], c_features[:, 1], label=f"Poisoned ({c})", marker='^', s=8, color=class_colors[i], alpha=0.7)
        else:
            plt.scatter(features_poisoned[:, 0], features_poisoned[:, 1], label="Poisoned", marker='^', s=8, color="red", alpha=0.7)
            
        plt.axis("off")
        plt.tight_layout()

        if save_dst:
            os.makedirs(save_dst, exist_ok=True)

            # Add enabled visual options to filename
            opts = np.array([highlight_cross, groupby_gt])
            opts_str = np.array(["highlight_cross", "groupby_gt"])[opts]
            if len(opts_str) > 0:
                filename = f"{'_'.join(opts_str)}"
            else:
                filename = "default"

            plt.savefig(os.path.join(save_dst, filename), transparent=True) 

        if show_plot:
            plt.legend()
            plt.show()
            plt.clf()
        else:
            plt.close()
    
    if show_plot or save_dst:
        # Create plots with various visual options
        for highlight_cross, groupby_gt in product([False, True], [False, True]):
            # Do not create plots highlighting cross samples if there are none
            if highlight_cross and not np.any(trainset.cross_lookup):
                continue

            # Do not create plots highlighting each original classes of poisoned samples if there are too many classes
            if groupby_gt and len(trainset.classes) != 10:
                continue

            plot(features_benign, features_poisoned, features_cross, gt_labels, highlight_cross, groupby_gt, show_plot, save_dst)

    # Concatenate benign and cross features, since we consider cross samples to be benign
    return np.concatenate([features_benign, features_cross]), features_poisoned, gt_labels

def clustering_score(features_benign, features_poisoned, silhouette=True):
    n_benign = len(features_benign)
    n_poisoned = len(features_poisoned)
    cluster_labels = np.concatenate([np.zeros(n_benign), np.ones(n_poisoned)])
    features = np.concatenate([features_benign, features_poisoned])
    
    if silhouette:
        return silhouette_score(features, cluster_labels)
    else:
        return davies_bouldin_score(features, cluster_labels)

SS = lambda x, y: clustering_score(x, y, silhouette=True)


# ----------------- Class-specific Davies-Bouldin Index (DBI)

# Average Davies-Bouldin Index between the benign cluster and poisoned subcluster for each class
def CDBI(features_benign, features_poisoned, gt_labels_poisoned):
    total = 0
    i = 0
    n_classes = N_CLASSES_DICT[DATASET]
    
    for c in range(n_classes):    
        c_indices = gt_labels_poisoned == c

        # Skip classes for which there are no poisoned samples
        # This happens for DFST (does not poison samples already of target class) and the clean-label attacks
        if not np.any(c_indices):
            continue

        total += clustering_score(features_benign, features_poisoned[c_indices], silhouette=False)
        i += 1

    return total / i





# ----------------- Discriminant Sliced-Wasserstein Distance (DWSD)
from ot import wasserstein_1d
from sklearn.preprocessing import normalize

def DSWD(model, feature_path, skip_misclassified=False):
    final_layer_name = "linear" if MODEL_ARCH == "resnet18" else "classifier"
    if MODEL_ARCH == 'vit_small':
        final_layer_name = "head"
    weights = model.state_dict()[f"{final_layer_name}.weight"].cpu()

    # Load features from the specified file
    feature_dict = torch.load(feature_path, weights_only=False)
    features_clean = feature_dict["features_clean"]
    features_bd = feature_dict["features_bd"]
    predictions = feature_dict["predictions_bd"]
    
    # Optionally filter features corresponding to wrongly classified inputs
    if skip_misclassified:
        correctly_classified = predictions == torch.full_like(predictions, TARGET_CLASS)
        features_clean = features_clean[correctly_classified]
        features_bd = features_bd[correctly_classified]

    return DSWD_eq_7(features_clean, features_bd, weights)

def DSWD_eq_7(features_clean, features_bd, param_matrix):
    """Implementation of Equation (7) of \"Backdoor Attack with Imperceptible Input and Latent Modification\".
    Based on the unofficial implementation from https://github.com/RJ-T/Wasserstein-Backdoor/blob/master/lira_trigger_generation.py.
    
    Notation from the paper: 
     - N = dataset size;
     - d = dimension of the latent space;
     - C = set of classes.

    :param features_clean: N x d matrix F_c
    :param features_bd: N x d matrix F_b
    :param param_matrix: |C| x d matrix W between the penultimate and the output layers
    """

    # Normalize each row of the parameter matrix using L2 norm
    param_matrix_norm = normalize(param_matrix, axis=1)
    
    # Transpose features to multiply them with rows of the parameter matrix
    features_clean = features_clean.T
    features_bd = features_bd.T
    projected_features_clean = np.matmul(param_matrix_norm, features_clean.numpy())
    projected_features_bd = np.matmul(param_matrix_norm, features_bd.numpy())

    # Summation in Equation 7
    classes = range(len(param_matrix))
    sum = np.sum([wasserstein_1d(projected_features_clean[c], projected_features_bd[c], p=2) for c in classes]) # wasserstein_1d(p=2) returns Wasserstein-2 distance raised to the power of 2
    
    # Divide by |C| and take square root
    dswd = (sum / len(classes))**0.5

    return dswd







# ----------------- Parameter-space
def UCLC(model, normalize=True): 
    """Based on the official implementation of Channel Lipschitzness-based Pruning: 
    https://github.com/rkteddy/channel-Lipschitzness-based-pruning/blob/main/defense.py."""
    
    uclc = torch.Tensor([])
    
    for name, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            std = m.running_var.sqrt()
            weight = m.weight

            channel_lips = []
            for idx in range(weight.shape[0]):
                # Combining weights of convolutions and BN
                w = conv.weight[idx].reshape(conv.weight.shape[1], -1) * (weight[idx]/std[idx]).abs()
                channel_lips.append(torch.svd(w.cpu())[1].max())
            channel_lips = torch.Tensor(channel_lips)

            if normalize:
                channel_lips = (channel_lips - channel_lips.mean()) / channel_lips.std()

            uclc = torch.cat((uclc, channel_lips))
        
       # Convolutional layer should be followed by a BN layer by default
        elif isinstance(m, nn.Conv2d):
            conv = m

    # Return the maximum deviation factor, as this indicates the channel with highest sensitivity, which is likely backdoored
    return uclc







# ----------------- 

# Trigger-Activated Change (TAC)

def TAC(tac_path, bd_or_clean="bd", skip_misclassified=False):
    # Load input-wise TAC from the specified file
    tac_dict = torch.load(tac_path, weights_only=False)
    tac_per_input = tac_dict[bd_or_clean]["tac"]
    predictions = tac_dict[bd_or_clean]["predictions_bd"]

    if skip_misclassified:
        correctly_classified = predictions == torch.full_like(predictions, TARGET_CLASS)
        tac_per_input = tac_per_input[correctly_classified]

    # Calculate average over all inputs
    return tac_per_input.mean(dim=0)




# ----------------- TAC-UCLC Product

def TUP(tac_path, model_bd):
    # Average TAC over all inputs for the backdoored and benign model
    tac_bd = TAC(tac_path, bd_or_clean="bd")
    tac_clean = TAC(tac_path, bd_or_clean="clean")

    # Sort the backdoored and benign model's TAC separately. remember the sorting indices for the former model
    bd_indices = np.argsort(tac_bd)
    tac_bd = tac_bd[bd_indices]
    tac_clean = np.sort(tac_clean)

    # Calculate the ratio between backdoored and benign model TAC
    tac_ratio = tac_bd / tac_clean

    # TAC is based on the batch normalization layer before the adaptive average pooling layer
    # To calculate UCLC on the same layer, we additionally require the convolutional layer before it
    if MODEL_ARCH == "resnet18":
        model_bd = model_bd.layer4[1] 
        model_bd = nn.Sequential(model_bd.conv2, model_bd.bn2)
    else:
        model_bd = model_bd.features[-5:-3]

    # Calculate UCLC of the backdoored model and put neurons in the same order as the backdoored TAC values 
    uclc_bd = UCLC(model_bd, normalize=False)[bd_indices]

    return np.average(uclc_bd * tac_ratio)





# ----------------- Benign/backdoored data and models


# Add absolute path to ./backdoorbench to the system PATH in order to import backdoorbench functions
bb_dir = os.path.abspath("./backdoorbench")
sys.path.append(bb_dir)
import timm
from backdoorbench.models.resnet import ResNet18
from backdoorbench.models.vgg import VGG16
from backdoorbench.utils.save_load_attack import load_attack_result

def get_dataset(dataset, train=True, transforms=None):
    if dataset != 'tiny':
        data_root = os.path.join(DATA_DIR, dataset)

    if dataset == "cifar10":
        return torchvision.datasets.CIFAR10(root=data_root, 
                                            train=train, 
                                            download=True,
                                            transform=transforms)
    elif dataset == "cifar100":
        return torchvision.datasets.CIFAR100(root=data_root, 
                                             train=train, 
                                             download=True,
                                             transform=transforms)
    elif dataset == "tiny":
        if train:
            split = "train"
        else:
            split = "val"
        return torchvision.datasets.ImageFolder(
            os.path.join(DATA_DIR, "tiny-imagenet-200", split), transform=transforms)
    else:
        return Imagenette(root=data_root, 
                          train=train,
                          transform=transforms)

def load_model_state(arch, dataset, state_dict):
    if arch == "resnet18":
        model = ResNet18(num_classes=N_CLASSES_DICT[dataset]).to(DEVICE, non_blocking=True)
    elif arch == "vgg16":
        model = VGG16(num_classes=N_CLASSES_DICT[dataset]).to(DEVICE, non_blocking=True)
    elif arch == "vit_small":
        model = timm.create_model('vit_small_patch16_224', 
                num_classes=10, 
                patch_size=4, 
                img_size=32)
    else:
        raise Exception("Architecture not supported")
    
    model.load_state_dict(state_dict)
    model.eval()
    return model.to(DEVICE, non_blocking=True)

def load_clean_record(dataset, arch):
    record = {}

    # Get datasets
    for key in ["train", "test", "train_transformed", "test_transformed"]:
        record[key] = get_dataset(dataset, train="train" in key, transforms=TRANSFORM_DICT[f"{dataset}_{key}"])

        # Filter target class out of test datasets
        if key in ["test", "test_transformed"]:
            record[key] = filter_target_class(record[key], TARGET_CLASS)

    # Load dictionary containing model state dict
    clean_path = os.path.join(RECORD_DIR,
                              f"prototype_{experiment_variable_identifier(arch, dataset, None)}",
                              "clean_model.pth")
    state_dict = torch.load(clean_path)
    record["model"] = load_model_state(arch, dataset, state_dict)

    return record

def load_backdoor_record(dataset, arch, atk, poison_rate, clean_record):
    atk_path = os.path.join(RECORD_DIR,
                            f"{atk}_{experiment_variable_identifier(arch, dataset, poison_rate)}")

    # Different attack implementations use different functions 
    if atk in ["badnet", "blended", "wanet", "bpp", "narcissus"]:
        return load_backdoorbench(atk, atk_path, dataset, arch)
    elif atk in ["adaptive_patch", "adaptive_blend", ]:
        return load_adap(atk_path, dataset, arch, clean_record)
    elif atk == "dfst":
        return load_dfst(atk_path, dataset, arch, clean_record)
    elif atk == "grond":
        return load_grond(atk_path, dataset, arch)
    elif atk == "dfba":
        return load_dfba(atk_path, dataset, arch, clean_record)
    else:
        raise Exception(f"{atk} is not supported")
    
def load_backdoorbench(atk, atk_path, dataset, arch):
    record = {}
    atk_result = load_attack_result(os.path.join(atk_path, "attack_result.pt"))

    # Load backdoored data
    for key in ["train", "test"]:
        key_until_underscore = key.split('_')[0]
        bd_dataset = copy.deepcopy(atk_result[f"bd_{key_until_underscore}"])
        
        # Only narcissus needs an untransformed trainset to compare its asymmetric trigger
        # if key == "train" and atk != "narcissus":
        #     continue
        
        # replace_transform removes the existing normalization for train and test keys
        record[key] = BackdoorBenchDataset(bd_dataset, TARGET_CLASS, 
                                           replace_transform=TRANSFORM_DICT[f"{dataset}_{key}"])

    record["model"] = load_model_state(arch, dataset, atk_result["model"])

    return record

def load_adap(atk_path, dataset, arch, clean_record):
    record = {}

    # Load backdoored data
    for key in ["test", "train"]:
        key_until_underscore = key.split('_')[0]
        record[key] = AdapDataset(atk_path, TARGET_CLASS, 
                                  key_until_underscore, clean_record[key])
    
    state_dict = torch.load(os.path.join(atk_path, "model.pt"), map_location=DEVICE)
    record["model"] = load_model_state(arch, dataset, state_dict)

    return record

def load_dfst(atk_path, dataset, arch, clean_record):
    record = {}

    # Load backdoored data
    for key in ["test", "train"]:
        key_until_underscore = key.split('_')[0]
        record[key] = DFSTDataset(atk_path, TARGET_CLASS, 
                                  key_until_underscore, clean_record[key])

    state_dict = torch.load(os.path.join(atk_path, "model.pt"), map_location=DEVICE, weights_only=False)
    record["model"] = load_model_state(arch, dataset, state_dict)

    return record

def load_dfba(atk_path, dataset, arch, clean_record):
    record = {}

    # Load trigger, consisting of a mask and perturbation delta
    mask = torch.load(os.path.join(atk_path, "mask.pth"), weights_only=False)
    delta = torch.load(os.path.join(atk_path, "delta.pth"), weights_only=False)
    
    # Load backdoored test data (no train data available since attack is data-free)
    for key in ["test"]:
        record[key] = DFBADataset(clean_record[key], TARGET_CLASS, 
                                  delta, mask)
    
    state_dict = torch.load(os.path.join(atk_path, "model.pth"), map_location=DEVICE)
    record["model"] = load_model_state(arch, dataset, state_dict)

    return record

def load_grond(atk_path, dataset, arch):
    record = {}

    # Load backdoored data
    for key in ["test", "train"]:
        record[key] = GrondDataset(dataset, TRANSFORM_DICT[f"{dataset}_{key}"], TARGET_CLASS, atk_path, "train" in key)

    checkpoint = torch.load(os.path.join(atk_path, "checkpoint.pth"), map_location=DEVICE)
    state_dict = checkpoint["model"]
    record["model"] = load_model_state(arch, dataset, state_dict)

    return record



# Create benign test dataset by excluding samples of target class
def filter_target_class(dataset, target_class):
    targets_ndarray = np.array(dataset.targets)
    non_target_class = targets_ndarray != target_class
    dataset.data = dataset.data[non_target_class] 
    dataset.targets = list(np.array(dataset.targets)[non_target_class])

    return dataset

# Feed inputs from dataset to the model in order to extract the predictions
def extract_preds(model, dataset):
    predictions = torch.tensor([])
    dl = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    for in_batch, _ in iter(dl):
        with torch.no_grad():
            pred_batch = model.forward(in_batch.to(DEVICE, non_blocking=True))
            predictions = torch.cat([predictions, pred_batch.cpu().argmax(dim=1)])

    return predictions

# Feed inputs from dataset to the model in order to extract the predictions and features of the specified layer
def extract_preds_and_features(model, dataset, penultimate=True):
    predictions = torch.tensor([])
    features = torch.tensor([])
    dl = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    for in_batch, _ in iter(dl):
        size = len(in_batch) # May be smaller than bs for final batch

        with torch.no_grad():
            # Get features from the penultimate layer
            if penultimate:
                if MODEL_ARCH.startswith("vit"):
                    feat = model.forward_features(in_batch.to(DEVICE, non_blocking=True))
                    feat_batch = feat[:, 0]
                    pred_batch = model.forward_head(feat)
                else:
                    pred_batch, feat_batch = model.forward(in_batch.to(DEVICE, non_blocking=True), return_features=True)
                    feat_batch = feat_batch.reshape(size, -1)
            else: # Get features from the layer used to measure TAC in "Towards Backdoor Stealthiness in Model Parameter Space": https://arxiv.org/pdf/2501.05928v1
                if MODEL_ARCH.startswith("vit"):
                    feat = model.forward_features(in_batch.to(DEVICE, non_blocking=True))
                    feat_batch = feat[:, 0]
                    pred_batch = model.forward_head(feat)
                else:
                    pred_batch, feat_batch = model.forward_all_features(in_batch.to(DEVICE, non_blocking=True))
                    feat_batch = feat_batch[-1]

            features = torch.cat([features, feat_batch.cpu()])
            predictions = torch.cat([predictions, pred_batch.cpu().argmax(dim=1)])

    return predictions, features

# Given the experiment variable values, create a string identifier to use when loading and saving experiment results
def experiment_variable_identifier(model_arch, dataset, poison_rate):
    if poison_rate == None:
        return f"{model_arch}_{dataset}_pNone"
    else:
        poison_rate_str = "-".join(str(poison_rate).split(".")) # replace . by -
        return f"{model_arch}_{dataset}_p{poison_rate_str}"
    
def list_intersection(a: list, b: list):
    return list(set(a) & set(b))

def dict_subset(dict, keys):
    return {k: v for k, v in dict.items() if k in keys}

# ----------------- Extract features
# Functions

def get_sample_size(eval_type):
    # Evaluate all samples on GPU
    if DEVICE.type == "cuda": 
        return None

    # Otherwise, evaluate an appropiately sized sample 
    if eval_type == "input":
        return 10
    elif eval_type == "model":
        return 100
    else:
        raise Exception("Evaluation type unknown")
    
def save_train_feature_space(atk_id, model, trainset, path, sample_size=None):
    # Only save trainset features of the target class
    original_target_label = trainset.original_labels == trainset.target_class
    target_label_indices = np.argwhere(np.logical_or(original_target_label, trainset.poison_lookup)).squeeze()
    print('------- target_label_indices: ', len(target_label_indices))
    if sample_size:
        target_label_indices = np.random.choice(target_label_indices, size=sample_size, replace=False)

    subset = torch.utils.data.Subset(trainset, target_label_indices)
    preds, features = extract_preds_and_features(model, subset)

    train_features = {
        "features": features,
        "predictions": preds,
        "indices": target_label_indices
    }
    torch.save(train_features, os.path.join(path, atk_id))


def save_clean_test_preds_all_labels(atk_id, model, testset_clean, path, sample_size=None):
    sample_indices = np.array(range(len(testset_clean)))
    
    if sample_size:
        sample_indices = np.random.choice(sample_indices, size=sample_size, replace=False)
        testset_clean = torch.utils.data.Subset(testset_clean, sample_indices)

    preds_clean = extract_preds(model, testset_clean)
   
    test_features = {
        "predictions_clean": preds_clean,
        "indices": sample_indices
    }
    torch.save(test_features, os.path.join(path, atk_id))

def save_test_feature_space(atk_id, model, testset_clean, testset_bd, path, sample_size=None):
    sample_indices = np.array(range(len(testset_clean)))
    
    if sample_size:
        sample_indices = np.random.choice(sample_indices, size=sample_size, replace=False)
        testset_clean = torch.utils.data.Subset(testset_clean, sample_indices)
        testset_bd = torch.utils.data.Subset(testset_bd, sample_indices)

    preds_clean, features_clean = extract_preds_and_features(model, testset_clean)
    preds_bd, features_bd = extract_preds_and_features(model, testset_bd)

    test_features = {
        "features_clean": features_clean,
        "features_bd": features_bd,
        "predictions_clean": preds_clean,
        "predictions_bd": preds_bd,
        "indices": sample_indices
    }
    torch.save(test_features, os.path.join(path, atk_id))

def save_tac_activations(atk_id, model_clean, model_bd, testset_clean, testset_bd, path, sample_size=None):
    sample_indices = np.array(range(len(testset_clean)))
    
    if sample_size:
        sample_indices = np.random.choice(sample_indices, size=sample_size, replace=False)
        testset_clean = torch.utils.data.Subset(testset_clean, sample_indices)
        testset_bd = torch.utils.data.Subset(testset_bd, sample_indices)

    test_features = {
        "indices": sample_indices
    }

    # We extract features both with a benign and backdoored model, in order to compare their TAC values
    for model, desc in zip([model_clean, model_bd], ["clean", "bd"]):
        # To measure TAC, we extract features from a layer other than the penultimate layer
        preds_clean, features_clean = extract_preds_and_features(model, testset_clean, penultimate=False)
        preds_bd, features_bd = extract_preds_and_features(model, testset_bd, penultimate=False)

        diff = features_clean - features_bd

        # diff dimensions = number of inputs X amount of neurons in layer X kernel height X kernel width
        tac_per_input = torch.norm(diff, dim=(2,3)) 

        test_features[desc] = {
            "tac": tac_per_input,
            "predictions_clean": preds_clean,
            "predictions_bd": preds_bd,
        }

    torch.save(test_features, os.path.join(path, atk_id))

def save_tsne(atk_id, trainset, feature_train_path, path):
    feature_train_path = os.path.join(feature_train_path, f"{atk_id}.pt")
    save_dir = os.path.join(path, atk_id)
    features_benign, features_poisoned, gt_labels_poisoned = create_tsne(trainset, feature_train_path, save_dst=save_dir)
    
    tsne = {
        "features_benign": features_benign,
        "features_poisoned": features_poisoned,
        "gt_labels_poisoned": gt_labels_poisoned
    }

    torch.save(tsne, os.path.join(save_dir, "embedding.pt"))

def init_table(attacks, metrics):
    table = pd.DataFrame(data=np.full((len(attacks), len(metrics)), None), index=attacks, columns=metrics)
    table.columns.name = "Attack"

    return table

def metric_over_batches(metric, dataset1, dataset2):
    dl1 = torch.utils.data.DataLoader(dataset1, batch_size=BATCH_SIZE, shuffle=False)
    dl2 = torch.utils.data.DataLoader(dataset2, batch_size=BATCH_SIZE, shuffle=False)
    metric_vals = []
    batch_weights = [] # Compute weighted average since last batch may be smaller
    
    for (in_batch1, _), (in_batch2, _) in zip(iter(dl1), iter(dl2)):
        assert(len(in_batch1) == len(in_batch2))
        batch_weights.append(len(in_batch1))
        metric_vals.append(metric(in_batch1, in_batch2))

    return np.average(metric_vals, weights=batch_weights)

def input_stealth_eval(cleanset, bd_dataset, pr, sample_size=None, split="test", key='badnet'):
    METRIC_DICT = {
        "l1": l1_distance,
        "l2": l2_distance,
        "linf": linf_distance,
        "MSE": MSE,
        "PSNR": PSNR,
        "SSIM": SSIM,
        "LPIPS": LPIPS,
        "IS": IS,
        "pHash": pHash,
        "SAM": SAM,
    }
    # cleanset = cleanset[split]

    print('cleanset: ', len(cleanset))
    print('bd_dataset: ', len(bd_dataset))
    dataset = cleanset
    dataset_bd = bd_dataset
    input_stealth_table = {}

    # # Make sure all poisoned datasets have the same poison indices
    if split == "train":
        # atk_baseline = attacks[0]
        poison_lookup_compare = bd_dataset.poison_lookup

        # for atk in attacks[1:]:
        #     trainset = bd_dataset
            
        #     if not np.all(trainset.poison_lookup == poison_lookup_compare):
        #         raise Exception(f"Not all attacks have the same poisoned indices")
            
        poison_indices = np.argwhere(poison_lookup_compare).squeeze()
    else: # All samples are poisoned in test dataset
        poison_indices = range(len(dataset))

    if sample_size:
        poison_indices = np.random.choice(poison_indices, size=sample_size, replace=False)
    
    dataset = torch.utils.data.Subset(dataset, poison_indices)

    if key == "prototype":
        dataset_bd = dataset
    elif key == "dfba":
        dataset_bd = torch.utils.data.Subset(bd_dataset, poison_indices)
    else:
        dataset_bd = torch.utils.data.Subset(bd_dataset, poison_indices)
        
    for metric, metric_func in METRIC_DICT.items():
        measurement = metric_over_batches(metric_func, dataset, dataset_bd)
        input_stealth_table[metric] = measurement

    return input_stealth_table

class Imagenette(torchvision.datasets.VisionDataset):
    """Custom torchvision implementation of the Imagenette dataset.

    Unlike the official torchvision implementation of this dataset, this class has a 'data' attribute, which contains tensors representing all images.
    This way, our code can handle Imagenette the same way as CIFAR-10 and CIFAR-100, which already have a data attribute in their official torchvision implementation."""

    def __init__(self, root=None, train=True, transform=None, target_transform=None):
        super().__init__(root, transform=transform, target_transform=target_transform)

        split = "train" if train else "val"
        img_folder = torchvision.datasets.ImageFolder(os.path.join(self.root, split))

        self.classes = ['tench', 'English springer', 'cassette player', 'chain saw', 'church', 'French horn', 'garbage truck', 'gas pump', 'golf ball', 'parachute']
        self.data = []
        self.targets = []

        for img_path, label in img_folder.samples:
            img = np.array(Image.open(img_path).convert("RGB"))

            if len(img.shape) != 3:
                print(img_path)
                
            self.data.append(img)
            self.targets.append(label)

        self.data = np.stack(self.data)

# remove the target class to have the same length as backdoored testset.
class CLEAN_TEST(Dataset):

    def __init__(self, dataset, dataset_name, seed=0, transform=None, exclude_target=True, target_cls=0):
        self.transform = dataset.transform
        self.cleanset = dataset
        self.targets = self.cleanset.targets
        self.classes = dataset.classes

        non_target_cls_ids = [i for i in range(len(self.cleanset.targets)) if self.cleanset.targets[i] != target_cls]

        if dataset_name in ["tiny", "imagenette"]:
            self.data = []

            for img_path, label in dataset.samples:
                img = np.array(Image.open(img_path).convert("RGB"))

                if len(img.shape) != 3:
                    print(img_path)
                    
                self.data.append(img)

            self.data = np.stack(self.data)
        else:
            self.data = dataset.data


        if exclude_target:
            # Imagenette and Tiny ImageNet have different attribute names than CIFAR10(0)
            if dataset_name in ["tiny", "imagenette"]:
                self.cleanset.samples = [self.cleanset.samples[i] for i in non_target_cls_ids]
                self.cleanset.imgs = [self.cleanset.imgs[i] for i in non_target_cls_ids]
                poison_target = np.repeat(target_cls, len(self.cleanset.samples), axis=0)
            else:
                self.data = self.cleanset.data[non_target_cls_ids, :, :, :]
                self.targets = [self.cleanset.targets[i] for i in non_target_cls_ids]
            

    def __getitem__(self, index):
        target = self.targets[index]
        # img = self.data[index]
        img = self.cleanset[index][0]

        # if self.transform is not None:
        #     img = self.transform(img)
        return img, target
    
    def __len__(self):
        return len(self.targets)



def test(model, test_loader):
    with torch.no_grad():
        number_corrects = 0
        number_samples = 0
        for i, (test_images_set, test_labels_set) in enumerate(test_loader):
            test_images_set = test_images_set.cuda()
            test_labels_set = test_labels_set.cuda()

            y_predicted = model(test_images_set)
            labels_predicted = y_predicted.argmax(axis=1)
            number_corrects += (labels_predicted == test_labels_set).sum().item()
            number_samples += test_labels_set.size(0)
        print(f'Overall accuracy {(number_corrects / number_samples) * 100}%')

# do not normalize and do not augmentation, because the input space is supposed to be evaluated on real images.
def inputspace_permodel():
    dataset = 'tiny'
    attack = 'badnet'
    pr = 0.05
    str_pr = str(pr).replace('.', '-')
    # str_pr = 'None'
    arch = 'resnet18'
    atk_path = os.path.join(RECORD_DIR, f"{attack}_{arch}_tiny_p{str_pr}") + "/"

    clean_trainset = get_dataset(dataset, train=True, transforms=None)
    clean_testset = get_dataset(dataset, train=False, transforms=None)
    clean_trainset = CLEAN_TEST(clean_trainset, dataset, transform=transform_test[dataset], exclude_target=False, target_cls=TARGET_CLASS)
    clean_testset = CLEAN_TEST(clean_testset, dataset, transform=transform_test[dataset], target_cls=TARGET_CLASS)

    clean_record = {}
    clean_record['train'] = clean_trainset
    clean_record['test'] = clean_testset

    if attack == 'grond':
        record = load_grond(atk_path, dataset, arch)
    elif attack in ['prototype', 'badnet', 'blended', 'wanet', 'bpp', 'narcissus']:
        record = load_backdoorbench(attack, atk_path, dataset, arch)
    elif attack in ['adaptive_patch', 'adaptive_blend']:
        record = load_adap(atk_path, dataset, arch, clean_record)
    elif attack == "dfst":
        record = load_dfst(atk_path, dataset, arch, clean_record)
    elif attack == "dfba":
        record = load_dfba(atk_path, dataset, arch, clean_record)


    # test_loader = DataLoader(clean_testset, batch_size=256, shuffle=False, num_workers=6)
    # poi_test_loader = DataLoader(record['test'], batch_size=256, shuffle=False, num_workers=6)

    # print('clean test, ', end='')
    # test(record['model'], test_loader)
    # print('poi test, ', end='')
    # test(record['model'], poi_test_loader)

    input_results = input_stealth_eval(clean_testset, record['test'], pr, sample_size=None, split="test", key=attack)
    print("input_results: ", input_results)


def feature_space_permodel():
    sample_size = 5000
    dataset = 'cifar10'
    attack = 'adaptive_patch'
    pr = 0.05
    str_pr = str(pr).replace('.', '-')
    if attack == 'dfba':
        str_pr = 'None'
    # str_pr = 'None'
    arch = 'vgg16'
    atk_path = os.path.join(RECORD_DIR, f"{attack}_{arch}_cifar10_p{str_pr}") + "/"

    print('atk_path: ', atk_path)

    clean_trainset = get_dataset(dataset, train=True, transforms=transform_train[dataset])
    clean_testset = get_dataset(dataset, train=False, transforms=transform_test[dataset])

    # to have .data for tiny (imagefolder)
    clean_trainset = CLEAN_TEST(clean_trainset, dataset, transform=None, exclude_target=False, target_cls=TARGET_CLASS)
    
    testset_clean_without_target_class = filter_target_class(clean_testset, TARGET_CLASS)


    clean_record = {}
    clean_record['train'] = clean_trainset
    clean_record['test'] = testset_clean_without_target_class

    if attack == 'grond':
        record = load_grond(atk_path, dataset, arch)
    elif attack in ['prototype', 'badnet', 'blended', 'wanet', 'bpp', 'narcissus']:
        record = load_backdoorbench(attack, atk_path, dataset, arch)
    elif attack in ['adaptive_patch', 'adaptive_blend']:
        record = load_adap(atk_path, dataset, arch, clean_record)
    elif attack == "dfst":
        record = load_dfst(atk_path, dataset, arch, clean_record)
    elif attack == "dfba":
        record = load_dfba(atk_path, dataset, arch, clean_record)

    print('cleanset: ', len(testset_clean_without_target_class))
    print('bd_dataset: ', len(record['test']))


    test_loader = DataLoader(testset_clean_without_target_class, batch_size=256, shuffle=False, num_workers=6)
    poi_test_loader = DataLoader(record['test'], batch_size=256, shuffle=False, num_workers=6)

    print('clean test, ', end='')
    test(record['model'], test_loader)
    print('poi test, ', end='')
    test(record['model'], poi_test_loader)

    print('\n \n ----------------------------- ')


    # benign reference model of the architecture/dataset under evaluation
    state_dict = torch.load(os.path.join(RECORD_DIR, 'prototype_vgg16_cifar10_pNone', 'clean_model.pth'))
    try:
        print("try load state_dict['model'].")
        model_clean = load_model_state(arch, dataset, state_dict['model'])
    except:
        print("try load state_dict.")
        model_clean = load_model_state(arch, dataset, state_dict)
    
    print('clean test, ', end='')
    test(model_clean, test_loader)


    dirname = f"{arch}_{dataset}"
    train_path = os.path.join(RESULT_DIR, "feature_space_train", dirname)
    tsne_path = os.path.join(RESULT_DIR, "tsne", dirname)
    test_path = os.path.join(RESULT_DIR, "feature_space_test", dirname)
    preds_path = os.path.join(RESULT_DIR, "predictions_test_all_labels", dirname)
    tac_path = os.path.join(RESULT_DIR, "tac_activations", dirname)


    # # DFBA is data-free, it does not have a poisoned trainset
    if attack == "dfba":
        model_bd = record["model"]
        testset_bd = record["test"]
        save_clean_test_preds_all_labels(f"{attack}.pt", model_bd, testset_clean_without_target_class, preds_path, sample_size=sample_size)
        save_test_feature_space(f"{attack}.pt", model_bd, testset_clean_without_target_class, testset_bd, test_path, sample_size=sample_size)
        save_tac_activations(f"{attack}.pt", model_clean, model_bd, testset_clean_without_target_class, testset_bd, tac_path, sample_size=sample_size)
        return 0
    
    
    model_bd = record["model"]
    trainset_bd = record["train"]
    testset_bd = record["test"]
    save_clean_test_preds_all_labels(f"{attack}_p{pr}.pt", model_bd, testset_clean_without_target_class, preds_path, sample_size=sample_size)
    save_train_feature_space(f"{attack}_p{pr}.pt", model_bd, trainset_bd, train_path, sample_size=sample_size)
    save_tsne(f"{attack}_p{pr}", trainset_bd, train_path, tsne_path)
    save_test_feature_space(f"{attack}_p{pr}.pt", model_bd, testset_clean_without_target_class, testset_bd, test_path, sample_size=sample_size)
    if arch != 'vit_small':
        save_tac_activations(f"{attack}_p{pr}.pt", model_clean, model_bd, testset_clean_without_target_class, testset_bd, tac_path, sample_size=sample_size)


def feature_stealth_eval():
    feature_results_ss = {}
    feature_results_dswd = {}

    attck_dict = {
        "badnet": "BadNets",
        "blended": "Blend",
        "wanet": "WaNet",
        "bpp": "BppAttack",
        "adaptive_patch": "Adap-Patch",
        "adaptive_blend": "Adap-Blend",
        # "dfst": "DFST",
        # "dfba": "DFBA",
        "narcissus": "Narcissus",
        "grond": "Grond"
    }

    attack_list = list(attck_dict.keys())
    pr1 = [0.05]
    pr2 = [0.05]

    dataset = 'cifar10'
    arch = 'vgg16'

    print("---- ", f"{arch}_{dataset}")

    dir_prefix = RECORD_DIR + '/'

    for i in range(len(attack_list)):
        for j in range(len(pr1)):
            atk = attack_list[i]
            pr = pr1[j]
            print(atk, pr)
            if atk == 'narcissus' or atk == 'grond':
                pr = pr2[j]

            str_pr = str(pr).replace('.', '-')

            # Cannot evaluate SS for DFBA since it does not have poisoned training data
            # if atk != "dfba":
            #     continue


            exp_id = f"{arch}_{dataset}"
            atk_id = atk if atk == "dfba" else f"{atk}_p{pr}"
            
            
            if atk != "dfba":
                print('--- SS,')
                tsne_path = os.path.join(RESULT_DIR, "tsne", exp_id, atk_id, "embedding.pt")
                tsne_dict = torch.load(tsne_path, weights_only=False)
                feature_results_ss[atk_id] = SS(tsne_dict["features_benign"], tsne_dict["features_poisoned"])

            if atk in ['prototype', 'badnet', 'blended', 'wanet', 'bpp', 'narcissus']:
                model_path = dir_prefix + f"{atk}_{arch}_{dataset}_p{str_pr}/" + "attack_result.pt" 
                state_dict = torch.load(model_path, map_location=DEVICE)
                model = load_model_state(arch, dataset, state_dict['model'])
            elif atk in ['adaptive_patch', 'adaptive_blend']:
                model_path = dir_prefix + f"{atk}_{arch}_{dataset}_p{str_pr}/" + "model.pt" 
                state_dict = torch.load(model_path, map_location=DEVICE)
                model = load_model_state(arch, dataset, state_dict)
            elif atk == "dfst":
                model_path = dir_prefix + f"{atk}_{arch}_{dataset}_p{str_pr}/" + "model.pt" 
                state_dict = torch.load(model_path, map_location=DEVICE)
                model = load_model_state(arch, dataset, state_dict)
            elif atk == "dfba":
                model_path = dir_prefix + f"{atk}_{arch}_{dataset}_pNone/" + "model.pth" 
                state_dict = torch.load(model_path, map_location=DEVICE)
                model = load_model_state(arch, dataset, state_dict)
            
            print('model path, ', model_path)

            print('--- DSWD,')
            feature_save_path = os.path.join(RESULT_DIR, "feature_space_test", exp_id, f"{atk_id}.pt")
            feature_results_dswd[atk_id] = DSWD(model, feature_save_path)
                
    print('feature_results_ss: ', feature_results_ss)
    print('feature_results_dswd: ', feature_results_dswd)


def get_model_pr_combinations(attacks):
    combinations = []
    pr1 = [0.05, 0.003]
    pr2 = [0.004, 0.003]

    for key in attacks:
        if key in ["dfba"]:
            combinations.append((key, None))
            continue

        for i in range(len(pr1)):
            if key in ['narcissus', 'grond']:
                pr = pr2[i]
            else:
                pr = pr1[i]
        
            combinations.append((key, pr))
    return combinations


def parameter_stealth_eval():
    attck_dict = {
        "badnet": "BadNets",
        "blended": "Blend",
        "wanet": "WaNet",
        "bpp": "BppAttack",
        "adaptive_patch": "Adap-Patch",
        "adaptive_blend": "Adap-Blend",
        "dfst": "DFST",
        "dfba": "DFBA",
        "narcissus": "Narcissus",
        "grond": "Grond"
    }
    attacks = list(attck_dict.keys())
    atk_pr_tuples = get_model_pr_combinations(attacks)
   
    dir_prefix = RECORD_DIR + '/'

    arch = 'resnet18'
    dataset = 'tiny'

    param_results = {}

    for i, (atk, pr) in enumerate(atk_pr_tuples):
        if atk == 'dfba':
            str_pr = 'None'
        else:
            str_pr = str(pr).replace('.', '-')

        attck_name = f"{atk}_{arch}_{dataset}_p{str_pr}"
        attck_folder = os.path.join(dir_prefix, attck_name)

        if atk in ['prototype', 'badnet', 'blended', 'wanet', 'bpp', 'narcissus']:
            model_path = attck_folder + "/attack_result.pt" 
            state_dict = torch.load(model_path, map_location=DEVICE)
            model = load_model_state(arch, dataset, state_dict['model'])
        elif atk in ['adaptive_patch', 'adaptive_blend']:
            model_path = attck_folder + "/model.pt" 
            state_dict = torch.load(model_path, map_location=DEVICE)
            model = load_model_state(arch, dataset, state_dict)
        elif atk == "dfst":
            model_path = attck_folder + "/model.pt" 
            state_dict = torch.load(model_path, map_location=DEVICE)
            model = load_model_state(arch, dataset, state_dict)
        elif atk == "dfba":
            model_path = attck_folder + "/model.pth" 
            state_dict = torch.load(model_path, map_location=DEVICE)
            model = load_model_state(arch, dataset, state_dict)


        atk_id = atk if atk == "dfba" else f"{atk}_p{pr}"

        # UCLC
        uclc = UCLC(model)
        param_results[atk_id+'_uclc'] = uclc.max().item()
        # parameter_stealth_table.at[rows[j], columns[i]] = uclc.max().item()

        # TAC
        exp_id = f"{arch}_{dataset}"
        
        feature_save_path = os.path.join(RESULT_DIR, "tac_activations", exp_id, f"{atk_id}.pt")
        tac = TAC(feature_save_path)
        param_results[atk_id+'_tac'] = tac.max().item()
        # parameter_stealth_table.at[rows[j], columns[i]] = tac.max().item()
            
    print(param_results)


def new_metric_eval():
    attck_dict = {
        "badnet": "BadNets",
        "blended": "Blend",
        "wanet": "WaNet",
        "bpp": "BppAttack",
        "adaptive_patch": "Adap-Patch",
        "adaptive_blend": "Adap-Blend",
        "dfst": "DFST",
        "dfba": "DFBA",
        "narcissus": "Narcissus",
        "grond": "Grond"
    }
    attacks = list(attck_dict.keys())
    atk_pr_tuples = get_model_pr_combinations(attacks)
   
    dir_prefix = RECORD_DIR + '/'

    arch = 'resnet18'
    dataset = 'tiny'

    param_results = {}

    for i, (atk, pr) in enumerate(atk_pr_tuples):
        if atk == 'dfba':
            str_pr = 'None'
        else:
            str_pr = str(pr).replace('.', '-')

        attck_name = f"{atk}_{arch}_{dataset}_p{str_pr}"
        attck_folder = os.path.join(dir_prefix, attck_name)

        if atk in ['prototype', 'badnet', 'blended', 'wanet', 'bpp', 'narcissus']:
            model_path = attck_folder + "/attack_result.pt" 
            state_dict = torch.load(model_path, map_location=DEVICE)
            model = load_model_state(arch, dataset, state_dict['model'])
        elif atk in ['adaptive_patch', 'adaptive_blend']:
            model_path = attck_folder + "/model.pt" 
            state_dict = torch.load(model_path, map_location=DEVICE)
            model = load_model_state(arch, dataset, state_dict)
        elif atk == "dfst":
            model_path = attck_folder + "/model.pt" 
            state_dict = torch.load(model_path, map_location=DEVICE)
            model = load_model_state(arch, dataset, state_dict)
        elif atk == "dfba":
            model_path = attck_folder + "/model.pth" 
            state_dict = torch.load(model_path, map_location=DEVICE)
            model = load_model_state(arch, dataset, state_dict)


        atk_id = atk if atk == "dfba" else f"{atk}_p{pr}"
        exp_id = f"{arch}_{dataset}"

        # CDBI
        if atk != 'dfba':
            tsne_path = os.path.join(RESULT_DIR, "tsne", exp_id, atk_id, "embedding.pt")
            tsne_dict = torch.load(tsne_path, weights_only=False)
            cdbi = CDBI(tsne_dict["features_benign"], 
                       tsne_dict["features_poisoned"], 
                       tsne_dict["gt_labels_poisoned"])
            param_results[atk_id+'cdbi'] = cdbi


        # TUP
        tac_save_path = os.path.join(RESULT_DIR, "tac_activations", exp_id, f"{atk_id}.pt")
        tup = TUP(tac_save_path, model)
        param_results[atk_id+'tup'] = tup
            
    print(param_results)



if __name__ == '__main__':
    new_metric_eval()
