
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image, ImageFilter, ImageOps
import random
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import autocast, GradScaler
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

###########################################
# 1. Define the hierarchical mappings
###########################################

subclass_mapping = {
    # Group 1: Basic Structural Elements
    # Superclass: Bricks
    # Class: Standard Bricks
    "3001": "2x4 Brick",
    "3002": "2x3 Brick",
    "3003": "2x2 Brick",

    # Class: Specialty Bricks
    "2357": "Corner Brick",
    "3622": "Arch Brick",
    "3009": "Sloped Brick",

    # Class: Large Bricks
    "3004": "2x6 Brick",
    "3005": "2x8 Brick",
    "3008": "2x10 Brick",

    # Superclass: Plates
    # Class: Flat Plates
    "3020": "2x4 Plate",
    "3021": "2x3 Plate",
    "3023": "2x2 Plate",

    # Class: Textured Plates
    "3832": "Baseplate",
    "3665": "Grille Plate",
    "3460": "Sloped Plate",

    # Class: Specialty Plates
    "3024": "1x1 Round Plate",
    "4477": "1x10 Plate",
    "60479": "1x12 Plate",

    # Superclass: Tiles
    # Class: Flat Tiles
    "2431": "2x2 Corner Tile",
    "6636": "1x6 Tile",
    "4162": "1x8 Tile",

    # Class: Decorative Tiles
    "63864": "Tile with Hole",
    "2436": "Grooved Tile",
    "3068b": "Inverted Tile",

    # Class: Functional Tiles
    "32278": "Slope Tile",
    "22388": "Curved Tile",
    "32523": "Rotation Tile",

    # Group 2: Functional Components
    # Superclass: Clips & Fasteners
    # Class: Horizontal Fasteners
    "48729": "Bar Holder",
    "60474": "Clip with Handle",
    "15712": "Double Clip",

    # Class: Vertical Fasteners
    "63868": "1x2 Clip",
    "60478": "Vertical Handle",
    "60470": "Pin Clip",

    # Class: Specialty Connectors
    "88072": "Angled Clip",
    "99207": "Bracket with Holes",
    "99780": "Robot Arm",

    # Superclass: Hinges & Axles
    # Class: Hinges
    "2429": "Hinge Plate",
    "2430": "Hinge Brick",
    "3937": "1x2 Hinge",

    # Class: Axles
    "3705": "3L Axle",
    "6536": "8L Axle",
    "60451": "16L Axle",

    # Class: Rotation Parts
    "3673": "Turntable",
    "32523": "Rotation Joint",   # duplicate key; only one instance kept
    "32524": "Axle Connector",

    # Superclass: Technic Parts
    # Class: Panels
    "32524": "5x5 Panel",  # duplicate key
    "32525": "Curved Panel",
    "32526": "Sloped Panel",

    # Class: Gears
    "53451": "24-Tooth Gear",
    "54200": "Bevel Gear",
    "60483": "Worm Gear",

    # Class: Axle Accessories
    "48336": "Axle Connector",
    "54201": "Axle with Stop",
    "60484": "Axle with Pin",

    # Group 3: Decorative & Specialty
    # Superclass: Minifigure Accessories
    # Class: Tools
    "15573": "Jumper Plate",
    "18651": "Paint Roller",
    "60471": "Pin Clip Tool",

    # Class: Food & Props
    "2586": "Drumstick",
    "79743": "Cupcake",
    "30229": "Zipline Handle",

    # Class: Weapons
    "27150": "Blade",
    "27167": "Treasure Chest Lid",
    "30230": "Weapon Handle",

    # Superclass: Nature Elements
    # Class: Plants
    "6854": "Fern",
    "50950": "Plant Wedge",
    "3565": "Plant Leaf",

    # Class: Rocks & Terrain
    "77822": "Ramp Loop",
    "35646": "Rock",
    "1998": "Terrain Piece",

    # Class: Animal Parts
    "25893": "Horse Hitching",
    "11203": "Trap Door",
    "50951": "Animal Wedge",

    # Superclass: Curved Parts
    # Class: Arches
    "3623": "1x3 Arch",
    "3224": "2x2 Arch",
    "32279": "Curved Slope",

    # Class: Cylinders
    "22885": "1x2x3 Cylinder",
    "43888": "1x1x6 Round",
    "22389": "Curved Cylinder",

    # Class: Wedge Plates
    "32278": "45° Wedge",   # duplicate key
    "22390": "Curved Wedge",
    "50950": "10° Wedge"     # duplicate key
}

class_mapping = {
    "2x4 Brick": "Standard Bricks", "2x3 Brick": "Standard Bricks", "2x2 Brick": "Standard Bricks",
    "Corner Brick": "Specialty Bricks", "Arch Brick": "Specialty Bricks", "Sloped Brick": "Specialty Bricks",
    "2x6 Brick": "Large Bricks", "2x8 Brick": "Large Bricks", "2x10 Brick": "Large Bricks",

    "2x4 Plate": "Flat Plates", "2x3 Plate": "Flat Plates", "2x2 Plate": "Flat Plates",
    "Baseplate": "Textured Plates", "Grille Plate": "Textured Plates", "Sloped Plate": "Textured Plates",
    "1x1 Round Plate": "Specialty Plates", "1x10 Plate": "Specialty Plates", "1x12 Plate": "Specialty Plates",

    "2x2 Corner Tile": "Flat Tiles", "1x6 Tile": "Flat Tiles", "1x8 Tile": "Flat Tiles",
    "Tile with Hole": "Decorative Tiles", "Grooved Tile": "Decorative Tiles", "Inverted Tile": "Decorative Tiles",
    "Slope Tile": "Functional Tiles", "Curved Tile": "Functional Tiles", "Rotation Tile": "Functional Tiles",

    "Bar Holder": "Horizontal Fasteners", "Clip with Handle": "Horizontal Fasteners", "Double Clip": "Horizontal Fasteners",
    "1x2 Clip": "Vertical Fasteners", "Vertical Handle": "Vertical Fasteners", "Pin Clip": "Vertical Fasteners",
    "Angled Clip": "Specialty Connectors", "Bracket with Holes": "Specialty Connectors", "Robot Arm": "Specialty Connectors",

    "Hinge Plate": "Hinges", "Hinge Brick": "Hinges", "1x2 Hinge": "Hinges",
    "3L Axle": "Axles", "8L Axle": "Axles", "16L Axle": "Axles",
    "Turntable": "Rotation Parts", "Rotation Joint": "Rotation Parts", "Axle Connector": "Rotation Parts",

    "5x5 Panel": "Panels", "Curved Panel": "Panels", "Sloped Panel": "Panels",
    "24-Tooth Gear": "Gears", "Bevel Gear": "Gears", "Worm Gear": "Gears",
    "Axle Connector": "Axle Accessories", "Axle with Stop": "Axle Accessories", "Axle with Pin": "Axle Accessories",

    "Jumper Plate": "Tools", "Paint Roller": "Tools", "Pin Clip Tool": "Tools",
    "Drumstick": "Food & Props", "Cupcake": "Food & Props", "Zipline Handle": "Food & Props",
    "Blade": "Weapons", "Treasure Chest Lid": "Weapons", "Weapon Handle": "Weapons",

    "Fern": "Plants", "Plant Wedge": "Plants", "Plant Leaf": "Plants",
    "Ramp Loop": "Rocks & Terrain", "Rock": "Rocks & Terrain", "Terrain Piece": "Rocks & Terrain",
    "Horse Hitching": "Animal Parts", "Trap Door": "Animal Parts", "Animal Wedge": "Animal Parts",

    "1x3 Arch": "Arches", "2x2 Arch": "Arches", "Curved Slope": "Arches",
    "1x2x3 Cylinder": "Cylinders", "1x1x6 Round": "Cylinders", "Curved Cylinder": "Cylinders",
    "45° Wedge": "Wedge Plates", "Curved Wedge": "Wedge Plates", "10° Wedge": "Wedge Plates"
}

superclass_mapping = {
    "Standard Bricks": "Bricks", "Specialty Bricks": "Bricks", "Large Bricks": "Bricks",
    "Flat Plates": "Plates", "Textured Plates": "Plates", "Specialty Plates": "Plates",
    "Flat Tiles": "Tiles", "Decorative Tiles": "Tiles", "Functional Tiles": "Tiles",

    "Horizontal Fasteners": "Clips & Fasteners", "Vertical Fasteners": "Clips & Fasteners", "Specialty Connectors": "Clips & Fasteners",
    "Hinges": "Hinges & Axles", "Axles": "Hinges & Axles", "Rotation Parts": "Hinges & Axles",
    "Panels": "Technic Parts", "Gears": "Technic Parts", "Axle Accessories": "Technic Parts",

    "Tools": "Minifigure Accessories", "Food & Props": "Minifigure Accessories", "Weapons": "Minifigure Accessories",
    "Plants": "Nature Elements", "Rocks & Terrain": "Nature Elements", "Animal Parts": "Nature Elements",
    "Arches": "Curved Parts", "Cylinders": "Curved Parts", "Wedge Plates": "Curved Parts"
}

group_mapping = {
    "Bricks": "Basic Structural Elements",
    "Plates": "Basic Structural Elements",
    "Tiles": "Basic Structural Elements",

    "Clips & Fasteners": "Functional Components",
    "Hinges & Axles": "Functional Components",
    "Technic Parts": "Functional Components",

    "Minifigure Accessories": "Decorative & Specialty",
    "Nature Elements": "Decorative & Specialty",
    "Curved Parts": "Decorative & Specialty"
}

# ===============================
# 2. Create Node Sets for Each Hierarchical Level
# ===============================

# Level 1: Groups – use the unique group names from group_mapping (values)
groups = sorted(set(group_mapping.values()))

# Level 2: Superclasses – use the keys of group_mapping
superclasses = sorted(set(group_mapping.keys()))

# Level 3: Classes – use unique values from class_mapping
classes = sorted(set(class_mapping.values()))

# Level 4: Subclasses – use unique values from subclass_mapping
subclasses = sorted(set(subclass_mapping.values()))

# Level 5: Parts – use keys of subclass_mapping
parts = sorted(set(subclass_mapping.keys()))

print("Level 1 (Groups):", groups)
print("Level 2 (Superclasses):", superclasses)
print("Level 3 (Classes):", classes)
print("Level 4 (Subclasses):", subclasses)
print("Level 5 (Parts) [sample]:", parts[:10], "...")   # show a sample

# ===============================
# 3. Build Unified Node List and Lookup
# ===============================

# Nodes appear in order: Groups, then Superclasses, then Classes, then Subclasses, then Parts
all_nodes = groups + superclasses + classes + subclasses + parts
node_to_idx = {node: idx for idx, node in enumerate(all_nodes)}
n_total = len(all_nodes)
print("Total number of unique nodes in unified hierarchy:", n_total)

# ===============================
# 4. Create the Unified Adjacency Matrix
# ===============================
# Build a matrix A_unified (n_total x n_total)
A_unified = np.zeros((n_total, n_total), dtype=int)

# (a) Group -> Superclass (using group_mapping)
for sc in superclasses:
    group = group_mapping.get(sc, None)
    if group is not None:
        parent_idx = node_to_idx[group]
        child_idx = node_to_idx[sc]
        A_unified[parent_idx, child_idx] = 1

# (b) Superclass -> Class (using superclass_mapping)
for cls, sc in superclass_mapping.items():
    if sc in node_to_idx and cls in node_to_idx:
        parent_idx = node_to_idx[sc]
        child_idx = node_to_idx[cls]
        A_unified[parent_idx, child_idx] = 1

# (c) Class -> Subclass (using class_mapping)
for subcls, cls in class_mapping.items():
    if cls in node_to_idx and subcls in node_to_idx:
        parent_idx = node_to_idx[cls]
        child_idx = node_to_idx[subcls]
        A_unified[parent_idx, child_idx] = 1

# (d) Subclass -> Part (using subclass_mapping)
for part, subcls in subclass_mapping.items():
    if subcls in node_to_idx and part in node_to_idx:
        parent_idx = node_to_idx[subcls]
        child_idx = node_to_idx[part]
        A_unified[parent_idx, child_idx] = 1

###########################################
# 5. Ancestry and Target Vector
###########################################
def get_ancestry_path(A, leaf_idx):
    """Get the ancestry path from leaf to root in the hierarchy"""
    path = [leaf_idx]
    current = leaf_idx
    while True:
        parents = np.where(A[:, current] == 1)[0]
        if len(parents) == 0:
            break
        parent = parents[0]
        path.insert(0, parent)
        current = parent
    return path

def get_target_vector(group, supercls, cls, subcls, part, all_nodes, node_to_idx, A_unified):
    """Generate a target vector based on the hierarchy path"""
    leaf_idx = node_to_idx[part]
    ancestry = get_ancestry_path(A_unified, leaf_idx)
    target = np.zeros(n_total, dtype=np.float32)
    target[ancestry] = 1.0
    return target

def get_node_level(node_idx, groups, superclasses, classes, subclasses, parts):
    """Determine which level a node belongs to"""
    if node_idx < len(groups):
        return "Group"
    elif node_idx < len(groups) + len(superclasses):
        return "Superclass"
    elif node_idx < len(groups) + len(superclasses) + len(classes):
        return "Class"
    elif node_idx < len(groups) + len(superclasses) + len(classes) + len(subclasses):
        return "Subclass"
    else:
        return "Part"

###########################################
# 6. Data Augmentation
###########################################
class GaussianBlur:
    """Apply Gaussian blur with random radius"""
    def __init__(self, radius_min=0.1, radius_max=2.0):
        self.radius_min = radius_min
        self.radius_max = radius_max
        
    def __call__(self, img):
        radius = random.uniform(self.radius_min, self.radius_max)
        return img.filter(ImageFilter.GaussianBlur(radius=radius))

class RandomColorJitter:
    """Apply random color jitter with specified parameters"""
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        
    def __call__(self, img):
        transforms_list = []
        if self.brightness > 0:
            brightness_factor = random.uniform(max(0, 1 - self.brightness), 1 + self.brightness)
            transforms_list.append(lambda img: ImageOps.autocontrast(img.point(lambda x: min(255, max(0, brightness_factor * x)))))
        
        if self.contrast > 0:
            contrast_factor = random.uniform(max(0, 1 - self.contrast), 1 + self.contrast)
            transforms_list.append(lambda img: ImageEnhance.Contrast(img).enhance(contrast_factor))
        
        random.shuffle(transforms_list)
        img = img.copy()
        for transform in transforms_list:
            img = transform(img)
        return img

# Enhanced data augmentation transforms
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    GaussianBlur(radius_min=0.1, radius_max=1.5),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

###########################################
# 7. Dataset
###########################################
class HierarchicalLegoDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.valid_paths = []  # To track which images were successfully loaded

        # Process each part directory
        for part in sorted(subclass_mapping.keys()):
            folder = os.path.join(root_dir, part)
            if os.path.isdir(folder):
                for fname in os.listdir(folder):
                    if fname.lower().endswith((".jpg", ".png", ".jpeg")):
                        img_path = os.path.join(folder, fname)
                        
                        # Verify the file is valid
                        try:
                            with Image.open(img_path) as img:
                                # Get all hierarchical labels
                                subcls = subclass_mapping[part]
                                cls_label = class_mapping.get(subcls, None)
                                if cls_label is None: continue
                                supercls = superclass_mapping.get(cls_label, None)
                                if supercls is None: continue
                                group = group_mapping.get(supercls, None)
                                if group is None: continue
                                
                                self.image_paths.append(img_path)
                                self.labels.append((group, supercls, cls_label, subcls, part))
                                self.valid_paths.append(img_path)
                        except:
                            print(f"Warning: Could not open {img_path}")
                            continue

        print(f"Loaded {len(self.image_paths)} valid images out of {len(self.valid_paths)} attempted.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        label_tuple = self.labels[index]
        
        try:
            img = Image.open(img_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            
            target = get_target_vector(*label_tuple, all_nodes, node_to_idx, A_unified)
            target = torch.tensor(target)
            return img, target
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a placeholder image and target if there's an error
            placeholder_img = torch.zeros(3, 224, 224)
            placeholder_target = torch.zeros(len(all_nodes))
            return placeholder_img, placeholder_target

###########################################
# 8. Advanced Hierarchical Model
###########################################
class HierarchicalAttentionModel(nn.Module):
    def __init__(self, target_dim, hierarchy_levels=5):
        super(HierarchicalAttentionModel, self).__init__()
        
        # Use EfficientNet-B0 as the backbone
        self.backbone = models.efficientnet_b0(pretrained=True)
        num_ftrs = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        
        # Feature attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(num_ftrs, num_ftrs // 2),
            nn.ReLU(),
            nn.Linear(num_ftrs // 2, num_ftrs),
            nn.Sigmoid()
        )
        
        # Hierarchical feature processing
        self.fc_shared = nn.Linear(num_ftrs, 512)
        self.bn_shared = nn.BatchNorm1d(512)
        
        # Level-specific processing branches
        self.level_embeddings = nn.Embedding(hierarchy_levels, 64)
        
        # Level-specific layers with different dropout rates
        self.level_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512 + 64, 256),  # +64 for level embedding
                nn.ReLU(),
                nn.Dropout(0.2 + i * 0.05),  # Increasing dropout by level
                nn.Linear(256, target_dim)
            ) for i in range(hierarchy_levels)
        ])
        
        # Initialize weights properly
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights for better convergence"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        # Extract features from backbone
        features = self.backbone(x)
        
        # Apply attention mechanism
        att_weights = self.attention(features)
        weighted_features = features * att_weights
        
        # Shared representation
        shared_feats = self.fc_shared(weighted_features)
        shared_feats = self.bn_shared(shared_feats)
        shared_feats = F.relu(shared_feats)
        
        # Apply level-specific processing
        outputs = []
        batch_size = x.size(0)
        
        for i, level_layer in enumerate(self.level_layers):
            # Get level embedding
            level_idx = torch.full((batch_size,), i, dtype=torch.long, device=x.device)
            level_embedding = self.level_embeddings(level_idx)
            
            # Concatenate features with level embedding
            level_input = torch.cat([shared_feats, level_embedding], dim=1)
            
            # Apply level-specific layers
            level_output = level_layer(level_input)
            outputs.append(level_output)
        
        # Combine all level outputs with attention to hierarchy
        combined_output = sum(outputs) / len(outputs)
        return combined_output

###########################################
# 9. Hierarchical Loss Function
###########################################
class HierarchicalLoss(nn.Module):
    def __init__(self, hierarchy_matrix, alpha=1.0, beta=0.5, gamma=0.3, level_weights=None):
        super(HierarchicalLoss, self).__init__()
        self.hierarchy_matrix = hierarchy_matrix  # Adjacency matrix
        self.alpha = alpha  # Weight for direct predictions
        self.beta = beta    # Weight for ancestor consistency
        self.gamma = gamma  # Weight for level-specific loss
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        
        # Default level weights if not provided
        if level_weights is None:
            # Emphasize higher levels more
            self.level_weights = {
                "Group": 1.0,
                "Superclass": 0.9, 
                "Class": 0.8,
                "Subclass": 0.7,
                "Part": 0.6
            }
        else:
            self.level_weights = level_weights
        
    def forward(self, outputs, targets, node_levels):
        """
        Calculate hierarchical loss
        
        Args:
            outputs: Model predictions (batch_size, n_nodes)
            targets: Ground truth (batch_size, n_nodes)
            node_levels: Dictionary mapping node indices to their levels
            
        Returns:
            Total loss
        """
        # Standard BCE loss
        bce_loss = self.bce(outputs, targets)
        
        # Apply level-specific weighting
        weighted_bce_loss = torch.zeros_like(bce_loss)
        for i in range(bce_loss.shape[1]):
            level = node_levels.get(i, "Part")  # Default to Part if not found
            weighted_bce_loss[:, i] = bce_loss[:, i] * self.level_weights[level]
        
        # Get predicted probabilities
        probs = torch.sigmoid(outputs)
        
        # Consistency loss: ancestors should have higher probability than descendants
        consistency_loss = torch.zeros_like(weighted_bce_loss)
        for i in range(len(self.hierarchy_matrix)):
            for j in range(len(self.hierarchy_matrix)):
                if self.hierarchy_matrix[i, j] == 1:  # i is parent of j
                    # Parent prob should be >= child prob
                    consistency_loss[:, j] += F.relu(probs[:, j] - probs[:, i])
        
        # Combined loss
        total_loss = (self.alpha * weighted_bce_loss.mean() + 
                     self.beta * consistency_loss.mean())
        
        return total_loss

###########################################
# 10. Early Stopping
###########################################
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            verbose (bool): If True, prints a message for each validation loss improvement.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        
    def __call__(self, val_loss, model, path):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0
            
    def save_checkpoint(self, val_loss, model, path):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss

###########################################
# 11. Evaluation Functions
###########################################
def get_predicted_path(output, all_nodes, node_to_idx, A_unified, threshold=0.5):
    """Get the predicted path through the hierarchy based on thresholded predictions"""
    pred_indices = torch.where(output > threshold)[0].cpu().numpy().tolist()
    if not pred_indices:
        return []
    leaf = max(pred_indices, key=lambda idx: len(get_ancestry_path(A_unified, idx)))
    return get_ancestry_path(A_unified, leaf)

def compare_paths(pred_path, true_path):
    """Calculate path similarity as proportion of correctly predicted nodes"""
    min_len = min(len(pred_path), len(true_path))
    matches = sum(p == t for p, t in zip(pred_path[::-1][:min_len], true_path[::-1][:min_len]))
    return matches / max(len(true_path), 1)

def evaluate_hierarchical_accuracy(model, data_loader, device, all_nodes, node_to_idx, A_unified, 
                                  groups, superclasses, classes, subclasses, parts, threshold=0.5):
    """
    Evaluate hierarchical prediction accuracy at each level of the hierarchy
    
    Returns:
        Dictionary of accuracies at each level and overall path accuracy
    """
    model.eval()
    
    # Initialize metrics
    total_samples = 0
    correct_by_level = {
        "Group": 0,
        "Superclass": 0,
        "Class": 0, 
        "Subclass": 0,
        "Part": 0
    }
    path_similarity_sum = 0
    
    # Track node-level metrics
    node_metrics = {
        node_idx: {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
        for node_idx in range(len(all_nodes))
    }
    
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Evaluating"):
            images = images.to(device)
            targets = targets.to(device)
            batch_size = images.size(0)
            total_samples += batch_size
            
            # Get predictions
            outputs = model(images)
            predictions = (torch.sigmoid(outputs) > threshold).float()
            
            # Calculate metrics for each sample
            for i in range(batch_size):
                # Get true and predicted paths
                true_path = torch.where(targets[i] > 0.5)[0].cpu().numpy().tolist()
                pred_path = get_predicted_path(predictions[i], all_nodes, node_to_idx, A_unified, threshold)
                
                # Calculate path similarity
                if true_path:
                    similarity = compare_paths(pred_path, true_path)
                    path_similarity_sum += similarity
                
                # Get level-specific accuracy
                for level, level_name in enumerate(["Group", "Superclass", "Class", "Subclass", "Part"]):
                    level_indices = []
                    
                    # Define level indices based on the hierarchical structure
                    if level == 0:  # Group level
                        level_indices = list(range(len(groups)))
                    elif level == 1:  # Superclass level
                        level_indices = list(range(len(groups), len(groups) + len(superclasses)))
                    elif level == 2:  # Class level
                        level_indices = list(range(len(groups) + len(superclasses), 
                                               len(groups) + len(superclasses) + len(classes)))
                    elif level == 3:  # Subclass level
                        level_indices = list(range(len(groups) + len(superclasses) + len(classes),
                                               len(groups) + len(superclasses) + len(classes) + len(subclasses)))
                    elif level == 4:  # Part level
                        level_indices = list(range(len(groups) + len(superclasses) + len(classes) + len(subclasses),
                                               len(all_nodes)))
                    
                    # Check if prediction at this level is correct
                    true_nodes_at_level = set([idx for idx in true_path if idx in level_indices])
                    pred_nodes_at_level = set([idx for idx in pred_path if idx in level_indices])
                    
                    if true_nodes_at_level == pred_nodes_at_level and true_nodes_at_level:
                        correct_by_level[level_name] += 1
                
                # Update node-level metrics for precision/recall calculation
                for node_idx in range(len(all_nodes)):
                    pred_positive = node_idx in pred_path
                    true_positive = node_idx in true_path
                    
                    if pred_positive and true_positive:
                        node_metrics[node_idx]["TP"] += 1
                    elif pred_positive and not true_positive:
                        node_metrics[node_idx]["FP"] += 1
                    elif not pred_positive and true_positive:
                        node_metrics[node_idx]["FN"] += 1
                    else:  # not pred_positive and not true_positive
                        node_metrics[node_idx]["TN"] += 1
    
    # Calculate metrics
    level_accuracy = {level: correct / total_samples for level, correct in correct_by_level.items()}
    
    # Calculate mean path similarity
    mean_path_similarity = path_similarity_sum / total_samples
    
    # Calculate per-node precision, recall and F1
    node_performance = {}
    for node_idx, metrics in node_metrics.items():
        if metrics["TP"] + metrics["FP"] > 0:
            precision = metrics["TP"] / (metrics["TP"] + metrics["FP"])
        else:
            precision = 0
            
        if metrics["TP"] + metrics["FN"] > 0:
            recall = metrics["TP"] / (metrics["TP"] + metrics["FN"])
        else:
            recall = 0
            
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0
            
        node_name = all_nodes[node_idx]
        node_level = get_node_level(node_idx, groups, superclasses, classes, subclasses, parts)
        node_performance[node_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "level": node_level
        }
    
    # Compute average metrics by level
    level_metrics = {level: {"precision": 0, "recall": 0, "f1": 0, "count": 0} 
                    for level in ["Group", "Superclass", "Class", "Subclass", "Part"]}
    
    for node, perf in node_performance.items():
        level = perf["level"]
        level_metrics[level]["precision"] += perf["precision"]
        level_metrics[level]["recall"] += perf["recall"]
        level_metrics[level]["f1"] += perf["f1"]
        level_metrics[level]["count"] += 1
    
    for level, metrics in level_metrics.items():
        if metrics["count"] > 0:
            metrics["precision"] /= metrics["count"]
            metrics["recall"] /= metrics["count"]
            metrics["f1"] /= metrics["count"]
    
    return {
        "level_accuracy": level_accuracy,
        "path_similarity": mean_path_similarity,
        "level_metrics": level_metrics,
        "node_performance": node_performance
    }


###########################################
# 12. Training Loop and Model Management
###########################################
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                device, epochs=3, patience=7, checkpoint_path="best_model.pth",
                all_nodes=None, node_to_idx=None, A_unified=None, 
                groups=None, superclasses=None, classes=None, subclasses=None, parts=None):
    """
    Train the hierarchical model with validation and early stopping
    """
    # Initialize node level mapping for loss calculation
    node_levels = {}
    for idx in range(len(all_nodes)):
        node_levels[idx] = get_node_level(idx, groups, superclasses, classes, subclasses, parts)
    
    # Initialize early stopping
    early_stopping = EarlyStopping(patience=patience, verbose=True, delta=0.0005)
    
    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler()
    
    # Initialize history dictionary to track metrics
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': {level: [] for level in ["Group", "Superclass", "Class", "Subclass", "Part"]},
        'val_path_similarity': []
    }
    
    # Loop over epochs
    for epoch in range(epochs):
        start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            images = images.to(device)
            targets = targets.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Mixed precision forward pass
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, targets, node_levels)
            
            # Backward and optimize with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Update statistics
            train_loss += loss.item()
            train_batches += 1
        
        # Adjust learning rate
        scheduler.step()
        
        # Calculate average training loss
        avg_train_loss = train_loss / train_batches
        history['train_loss'].append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation"):
                images = images.to(device)
                targets = targets.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, targets, node_levels)
                
                val_loss += loss.item()
                val_batches += 1
        
        # Calculate average validation loss
        avg_val_loss = val_loss / val_batches
        history['val_loss'].append(avg_val_loss)
        
        # Evaluate hierarchical accuracy
        eval_metrics = evaluate_hierarchical_accuracy(
            model, val_loader, device, all_nodes, node_to_idx, A_unified,
            groups, superclasses, classes, subclasses, parts
        )
        
        # Update history with metrics
        for level, acc in eval_metrics['level_accuracy'].items():
            history['val_accuracy'][level].append(acc)
        
        history['val_path_similarity'].append(eval_metrics['path_similarity'])
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        # Print epoch summary
        print(f"Epoch {epoch+1}/{epochs} completed in {elapsed_time:.2f}s")
        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        print(f"Path Similarity: {eval_metrics['path_similarity']:.4f}")
        print("Level Accuracies:", {k: f"{v:.4f}" for k, v in eval_metrics['level_accuracy'].items()})
        
        # Check early stopping
        early_stopping(avg_val_loss, model, checkpoint_path)
        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break
    
    # Load best model before returning
    model.load_state_dict(torch.load(checkpoint_path))
    return model, history

###########################################
# 13. Plot Training Metrics
###########################################
def plot_training_history(history, save_path=None):
    """Plot training metrics history"""
    plt.figure(figsize=(15, 10))
    
    # Plot loss
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot path similarity
    plt.subplot(2, 2, 2)
    plt.plot(history['val_path_similarity'], label='Path Similarity', color='green')
    plt.title('Hierarchical Path Similarity')
    plt.xlabel('Epoch')
    plt.ylabel('Similarity')
    plt.grid(True, alpha=0.3)
    
    # Plot accuracy by level
    plt.subplot(2, 2, 3)
    for level, acc_list in history['val_accuracy'].items():
        plt.plot(acc_list, label=f'{level} Accuracy')
    plt.title('Accuracy by Hierarchical Level')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot accuracy comparison for latest epoch
    plt.subplot(2, 2, 4)
    latest_accuracies = {level: acc_list[-1] for level, acc_list in history['val_accuracy'].items()}
    levels = list(latest_accuracies.keys())
    values = list(latest_accuracies.values())
    
    plt.bar(levels, values, color='skyblue')
    plt.axhline(y=history['val_path_similarity'][-1], color='r', linestyle='--', label='Path Similarity')
    plt.title('Final Accuracy by Level')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    
    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f'{v:.3f}', ha='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

###########################################
# 14. Inference and Visualization
###########################################
def visualize_prediction(image_path, model, transform, all_nodes, node_to_idx, A_unified, threshold=0.5):
    """Visualize a hierarchical prediction for a single image"""
    # Load and transform image
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.sigmoid(output)[0]
    
    # Get predicted path
    pred_path = get_predicted_path(probabilities, all_nodes, node_to_idx, A_unified, threshold)
    
    # Create visualization
    plt.figure(figsize=(15, 10))
    
    # Display image
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Input Image")
    plt.axis('off')
    
    # Display hierarchical prediction
    plt.subplot(1, 2, 2)
    plt.axis('off')
    
    y_position = 0.9
    for idx in pred_path:
        node_name = all_nodes[idx]
        prob_value = probabilities[idx].item()
        level = get_node_level(idx, groups, superclasses, classes, subclasses, parts)
        
        # Color-code by level
        if level == "Group":
            color = 'darkblue'
            indent = 0
        elif level == "Superclass":
            color = 'blue'
            indent = 1
        elif level == "Class":
            color = 'green'
            indent = 2
        elif level == "Subclass":
            color = 'orange'
            indent = 3
        else:  # Part
            color = 'red'
            indent = 4
            
        # Add text with indentation based on level
        plt.text(0.1 + indent * 0.1, y_position, f"{node_name}: {prob_value:.3f}",
                fontsize=12, color=color)
        y_position -= 0.05
    
    plt.title("Hierarchical Prediction")
    plt.tight_layout()
    plt.show()
    
    return pred_path, probabilities

def predict_batch(model, data_loader, device, all_nodes, node_to_idx, A_unified, 
                 groups, superclasses, classes, subclasses, parts, threshold=0.5, num_samples=5):
    """Make predictions on a batch and show results for a few samples"""
    model.eval()
    all_preds = []
    all_targets = []
    all_images = []
    
    # Get a batch
    with torch.no_grad():
        for images, targets in data_loader:
            batch_size = images.size(0)
            images = images.to(device)
            outputs = model(images)
            predictions = torch.sigmoid(outputs)
            
            # Store results
            all_images.extend(images.cpu())
            all_preds.extend(predictions.cpu())
            all_targets.extend(targets.cpu())
            
            # Only need a few samples
            if len(all_images) >= num_samples:
                break
    
    # Visualize predictions for the selected samples
    fig, axes = plt.subplots(num_samples, 2, figsize=(15, 5*num_samples))
    
    for i in range(min(num_samples, len(all_images))):
        # Original image
        img = all_images[i].permute(1, 2, 0).numpy()
        # Denormalize
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        
        # Predictions and ground truth
        pred = all_preds[i]
        target = all_targets[i]
        
        # Get paths
        pred_path = get_predicted_path(pred, all_nodes, node_to_idx, A_unified, threshold)
        true_path = torch.where(target > 0.5)[0].numpy().tolist()
        
        # Display image
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"Sample {i+1}")
        axes[i, 0].axis('off')
        
        # Compare predictions vs ground truth
        axes[i, 1].axis('off')
        
        # Display predictions and true labels
        text_content = "Prediction vs Ground Truth:\n\n"
        
        # Group level entries by their respective levels
        level_nodes = {
            "Group": [],
            "Superclass": [],
            "Class": [],
            "Subclass": [],
            "Part": []
        }
        
        # Fill the nodes by level for both predicted and true paths
        all_path_indices = sorted(set(pred_path + true_path))
        for idx in all_path_indices:
            level = get_node_level(idx, groups, superclasses, classes, subclasses, parts)
            node_name = all_nodes[idx]
            in_pred = idx in pred_path
            in_true = idx in true_path
            
            status = ""
            if in_pred and in_true:
                status = "✓"  # Correctly predicted
            elif in_pred and not in_true:
                status = "+"  # False positive
            elif not in_pred and in_true:
                status = "-"  # False negative
                
            level_nodes[level].append((node_name, in_pred, in_true, status))
            
        # Build the text display for each level
        for level, nodes in level_nodes.items():
            if nodes:
                text_content += f"{level}:\n"
                for node_name, in_pred, in_true, status in nodes:
                    pred_marker = "P" if in_pred else " "
                    true_marker = "T" if in_true else " "
                    text_content += f"  {status} {node_name} [{pred_marker}|{true_marker}]\n"
                text_content += "\n"
                
        axes[i, 1].text(0.1, 0.9, text_content, fontsize=10, 
                      verticalalignment='top', transform=axes[i, 1].transAxes)
        
    plt.tight_layout()
    plt.show()

###########################################
# 15. Main Training Script
###########################################
def main():
    # Set random seeds for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set paths
    data_dir = "/scratch/m24csa026/DL-Project/64"
    output_dir = "output/"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create datasets
    full_dataset = HierarchicalLegoDataset(data_dir, transform=train_transform)
    
    # Split dataset
    dataset_size = len(full_dataset)
    train_size = int(0.7 * dataset_size)
    val_size = int(0.15 * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    # Override transforms for validation and test
    val_dataset.dataset.transform = val_transform
    test_dataset.dataset.transform = test_transform
    
    # Create data loaders
    batch_size = 32
    num_workers = 4
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    
    for x,y in train_loader:
        print(x.shape)
        print(y.shape)
        break

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    print(f"Dataset splits: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    # Initialize model
    model = HierarchicalAttentionModel(len(all_nodes), hierarchy_levels=5).to(device)
    
    # Set up level weights for loss function
    level_weights = {
        "Group": 1.0,      # More weight to higher levels
        "Superclass": 0.9,
        "Class": 0.8,
        "Subclass": 0.7,
        "Part": 0.6
    }
    
    # Initialize loss function
    criterion = HierarchicalLoss(
        hierarchy_matrix=A_unified,
        alpha=1.0,     # Standard BCE loss weight
        beta=0.5,      # Hierarchy consistency weight
        gamma=0.3,     # Level-specific weight
        level_weights=level_weights
    )
    
    # Initialize optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-5)
    
    # Train model
    checkpoint_path = os.path.join(output_dir, "best_model.pth")
    print("Model training started...")
    trained_model, history = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        device, epochs=30, patience=7, checkpoint_path=checkpoint_path,
        all_nodes=all_nodes, node_to_idx=node_to_idx, A_unified=A_unified,
        groups=groups, superclasses=superclasses, classes=classes, 
        subclasses=subclasses, parts=parts
    )
    
    # Plot and save training history
    plot_training_history(history, save_path=os.path.join(output_dir, "training_history.png"))
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_metrics = evaluate_hierarchical_accuracy(
        trained_model, test_loader, device, all_nodes, node_to_idx, A_unified,
        groups, superclasses, classes, subclasses, parts
    )
    
    print("Test Set Results:")
    print(f"Path Similarity: {test_metrics['path_similarity']:.4f}")
    print("Level Accuracies:", {k: f"{v:.4f}" for k, v in test_metrics['level_accuracy'].items()})
    
    # Display predictions on a few test samples
    print("\nVisualizing predictions on test samples...")
    predict_batch(
        trained_model, test_loader, device, all_nodes, node_to_idx, A_unified,
        groups, superclasses, classes, subclasses, parts, num_samples=5
    )
    
    # Save test results
    test_results = {
        "metrics": test_metrics,
        "timestamp": time.strftime("%Y%m%d-%H%M%S")
    }
    
    # Convert complex NumPy/Torch objects to Python native types for JSON serialization
    # This is a simplified version, may need more complex conversion depending on objects
    def convert_to_serializable(obj):
        if isinstance(obj, (np.ndarray, np.number)):
            return obj.item() if obj.size == 1 else obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        else:
            return obj
    
    # Save test results to JSON
    with open(os.path.join(output_dir, "test_results.json"), 'w') as f:
        import json
        json.dump(convert_to_serializable(test_results), f, indent=4)
    
    print(f"Results saved to {output_dir}")
    return trained_model

if __name__ == "__main__":
    main()