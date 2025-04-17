
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

MODEL_PATH = "best_model.pth"  # Your trained model
IMAGE_PATH = "custom.png"       # Image to classify

# 4. Preprocessing transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 5. Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HierarchicalAttentionModel(target_dim=len(subclass_mapping)).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# 6. Prediction function
def predict(image_path):
    # Load and preprocess image
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # Get prediction
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.sigmoid(output)[0].cpu().numpy()
    
    # Get hierarchy path
    pred_indices = np.where(probs > 0.5)[0]
    if len(pred_indices) == 0:
        return []
    
    leaf_idx = pred_indices[np.argmax([len(get_ancestry_path(A_unified, idx)) 
                                   for idx in pred_indices])]
    hierarchy_path = get_ancestry_path(A_unified, leaf_idx)
    
    return [(all_nodes[idx], probs[idx]) for idx in hierarchy_path]

# 7. Visualization function
def visualize_prediction(image_path, predictions):
    plt.figure(figsize=(15, 6))
    
    # Show image
    plt.subplot(1, 2, 1)
    img = Image.open(image_path)
    plt.imshow(img)
    plt.axis("off")
    
    # Show hierarchy
    plt.subplot(1, 2, 2)
    plt.axis("off")
    y_pos = 0.9
    colors = ['darkblue', 'blue', 'green', 'orange', 'red']
    
    for i, (node, prob) in enumerate(predictions):
        level = get_node_level(node_to_idx[node], 
                             groups, superclasses, 
                             classes, subclasses, parts)
        plt.text(0.1, y_pos, 
                f"{level}: {node}\nConfidence: {prob:.3f}",
                fontsize=12,
                color=colors[i])
        y_pos -= 0.15
    
    plt.title("Hierarchical Prediction")
    plt.tight_layout()
    plt.show()

# 8. Run prediction and visualization
if __name__ == "__main__":
    predictions = predict(IMAGE_PATH)
    print("Hierarchical Prediction:")
    for node, prob in predictions:
        print(f"{node}: {prob:.4f}")
    
    visualize_prediction(IMAGE_PATH, predictions)