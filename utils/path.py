import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# Only models and transforms are needed from torchvision now
from torchvision import transforms, models
from PIL import Image # Keep PIL
from torch.utils.data import Dataset, DataLoader, random_split # Need these for split
import random # Need for random sampling
# Keep matplotlib for potential visualization if needed later, but not strictly for printing paths
# import matplotlib.pyplot as plt
from tqdm import tqdm # Keep tqdm if used in helper functions, otherwise optional
import time # Keep time if used, otherwise optional

# --- Configuration ---
# !!! ----- SET THESE VALUES ----- !!!
MODEL_PATH = "output/best_model.pth"  # Path to your saved model weights
DATA_DIR = "/scratch/m24csa026/DL-Project/64" # Base directory of the dataset used for training
OUTPUT_DIR = "output_predictions/" # Optional: For saving any potential output files
SEED = 42 # The exact random seed used during the original train/val/test split
NUM_SAMPLES = 5 # Number of random test images to predict
PREDICTION_THRESHOLD = 0.5 # Threshold for sigmoid probabilities
# --- End Configuration ---

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

###########################################
# 1. Define the hierarchical mappings (Keep EXACTLY as used for training)
###########################################
# --- Copy the full subclass_mapping, class_mapping, ---
# --- superclass_mapping, group_mapping dictionaries here ---
subclass_mapping = {
    # Group 1: Basic Structural Elements
    # Superclass: Bricks
    # Class: Standard Bricks
    "3001": "2x4 Brick", "3002": "2x3 Brick", "3003": "2x2 Brick",
    # Class: Specialty Bricks
    "2357": "Corner Brick", "3622": "Arch Brick", "3009": "Sloped Brick",
    # Class: Large Bricks
    "3004": "2x6 Brick", "3005": "2x8 Brick", "3008": "2x10 Brick",
    # Superclass: Plates
    # Class: Flat Plates
    "3020": "2x4 Plate", "3021": "2x3 Plate", "3023": "2x2 Plate",
    # Class: Textured Plates
    "3832": "Baseplate", "3665": "Grille Plate", "3460": "Sloped Plate",
    # Class: Specialty Plates
    "3024": "1x1 Round Plate", "4477": "1x10 Plate", "60479": "1x12 Plate",
    # Superclass: Tiles
    # Class: Flat Tiles
    "2431": "2x2 Corner Tile", "6636": "1x6 Tile", "4162": "1x8 Tile",
    # Class: Decorative Tiles
    "63864": "Tile with Hole", "2436": "Grooved Tile", "3068b": "Inverted Tile",
    # Class: Functional Tiles
    "32278": "Slope Tile", "22388": "Curved Tile", "32523": "Rotation Tile",
    # Group 2: Functional Components
    # Superclass: Clips & Fasteners
    # Class: Horizontal Fasteners
    "48729": "Bar Holder", "60474": "Clip with Handle", "15712": "Double Clip",
    # Class: Vertical Fasteners
    "63868": "1x2 Clip", "60478": "Vertical Handle", "60470": "Pin Clip",
    # Class: Specialty Connectors
    "88072": "Angled Clip", "99207": "Bracket with Holes", "99780": "Robot Arm",
    # Superclass: Hinges & Axles
    # Class: Hinges
    "2429": "Hinge Plate", "2430": "Hinge Brick", "3937": "1x2 Hinge",
    # Class: Axles
    "3705": "3L Axle", "6536": "8L Axle", "60451": "16L Axle",
    # Class: Rotation Parts
    "3673": "Turntable", "32523": "Rotation Joint", "32524": "Axle Connector",
    # Superclass: Technic Parts
    # Class: Panels
    "32524": "5x5 Panel", "32525": "Curved Panel", "32526": "Sloped Panel",
    # Class: Gears
    "53451": "24-Tooth Gear", "54200": "Bevel Gear", "60483": "Worm Gear",
    # Class: Axle Accessories
    "48336": "Axle Connector", "54201": "Axle with Stop", "60484": "Axle with Pin",
    # Group 3: Decorative & Specialty
    # Superclass: Minifigure Accessories
    # Class: Tools
    "15573": "Jumper Plate", "18651": "Paint Roller", "60471": "Pin Clip Tool",
    # Class: Food & Props
    "2586": "Drumstick", "79743": "Cupcake", "30229": "Zipline Handle",
    # Class: Weapons
    "27150": "Blade", "27167": "Treasure Chest Lid", "30230": "Weapon Handle",
    # Superclass: Nature Elements
    # Class: Plants
    "6854": "Fern", "50950": "Plant Wedge", "3565": "Plant Leaf",
    # Class: Rocks & Terrain
    "77822": "Ramp Loop", "35646": "Rock", "1998": "Terrain Piece",
    # Class: Animal Parts
    "25893": "Horse Hitching", "11203": "Trap Door", "50951": "Animal Wedge",
    # Superclass: Curved Parts
    # Class: Arches
    "3623": "1x3 Arch", "3224": "2x2 Arch", "32279": "Curved Slope",
    # Class: Cylinders
    "22885": "1x2x3 Cylinder", "43888": "1x1x6 Round", "22389": "Curved Cylinder",
    # Class: Wedge Plates
    "32278": "45° Wedge", "22390": "Curved Wedge", "50950": "10° Wedge"
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
    "Bricks": "Basic Structural Elements", "Plates": "Basic Structural Elements", "Tiles": "Basic Structural Elements",
    "Clips & Fasteners": "Functional Components", "Hinges & Axles": "Functional Components", "Technic Parts": "Functional Components",
    "Minifigure Accessories": "Decorative & Specialty", "Nature Elements": "Decorative & Specialty", "Curved Parts": "Decorative & Specialty"
}

###########################################
# 2. Create Node Sets & Unified Hierarchy (Keep EXACTLY as used for training)
###########################################
groups = sorted(set(group_mapping.values()))
superclasses = sorted(set(group_mapping.keys()))
classes = sorted(set(class_mapping.values()))
subclasses = sorted(set(subclass_mapping.values()))
parts = sorted(set(subclass_mapping.keys()))

all_nodes = groups + superclasses + classes + subclasses + parts
node_to_idx = {node: idx for idx, node in enumerate(all_nodes)}
n_total = len(all_nodes)
print(f"Total unique nodes in hierarchy: {n_total}")

A_unified = np.zeros((n_total, n_total), dtype=int)
# Populate A_unified (same logic as in training script)
# (a) Group -> Superclass
for sc in superclasses:
    group = group_mapping.get(sc)
    if group and group in node_to_idx and sc in node_to_idx:
        A_unified[node_to_idx[group], node_to_idx[sc]] = 1
# (b) Superclass -> Class
for cls, sc in superclass_mapping.items():
    if sc in node_to_idx and cls in node_to_idx:
        A_unified[node_to_idx[sc], node_to_idx[cls]] = 1
# (c) Class -> Subclass
for subcls, cls in class_mapping.items():
    if cls in node_to_idx and subcls in node_to_idx:
        A_unified[node_to_idx[cls], node_to_idx[subcls]] = 1
# (d) Subclass -> Part
for part, subcls in subclass_mapping.items():
    if subcls in node_to_idx and part in node_to_idx:
        A_unified[node_to_idx[subcls], node_to_idx[part]] = 1

###########################################
# 3. Helper Functions (Keep EXACTLY as used for training/evaluation)
###########################################
def get_ancestry_path(A, leaf_idx):
    path = [leaf_idx]
    current = leaf_idx
    while True:
        parents = np.where(A[:, current] == 1)[0]
        if len(parents) == 0: break
        parent = parents[0]
        path.insert(0, parent)
        current = parent
    return path

# ***** ADDED THIS FUNCTION BACK *****
def get_target_vector(group, supercls, cls, subcls, part, all_nodes, node_to_idx, A_unified):
    """Generate a target vector based on the hierarchy path"""
    target = np.zeros(len(all_nodes), dtype=np.float32)
    try:
        # Ensure the part exists in the mapping
        if part not in node_to_idx:
             raise KeyError(f"Part ID '{part}' not found in node_to_idx mapping during target vector generation.")
        leaf_idx = node_to_idx[part]
        ancestry = get_ancestry_path(A_unified, leaf_idx)

        # Optional Sanity check: Ensure provided labels match derived ancestry
        # This helps catch inconsistencies in mappings or label tuples
        derived_nodes_set = set(all_nodes[i] for i in ancestry)
        expected_nodes_set = {group, supercls, cls, subcls, part}
        if not expected_nodes_set.issubset(derived_nodes_set):
             print(f"Warning: Provided GT labels {expected_nodes_set} mismatch derived ancestry {derived_nodes_set} for part {part}. Using derived ancestry for target.")
             # Proceed using the ancestry derived from the part ID as it's likely more reliable if mappings are correct

        target[ancestry] = 1.0
    except KeyError as e:
        print(f"Error generating target vector: {e}")
        # Return an all-zero vector in case of error
        target.fill(0)
    except Exception as e:
        print(f"Unexpected error in get_target_vector for part {part}: {e}")
        target.fill(0)
    return target
# **********************************

def get_node_level(node_idx, groups, superclasses, classes, subclasses, parts):
    # (Same logic as in training script)
    len_g = len(groups)
    len_sc = len_g + len(superclasses)
    len_c = len_sc + len(classes)
    len_s = len_c + len(subclasses)
    if node_idx < len_g: return "Group"
    elif node_idx < len_sc: return "Superclass"
    elif node_idx < len_c: return "Class"
    elif node_idx < len_s: return "Subclass"
    else: return "Part"

def get_predicted_path(output_logits, all_nodes, node_to_idx, A_unified, threshold=0.5):
    # (Same logic as in previous script)
    probabilities = torch.sigmoid(output_logits) # Convert logits to probabilities
    pred_indices = torch.where(probabilities > threshold)[0].cpu().numpy().tolist()
    if not pred_indices: return []

    deepest_node_idx = -1
    valid_leaves = []
    part_level_start_index = n_total - len(parts)

    # Identify potential leaf nodes among predictions (those at Part level)
    for idx in pred_indices:
        # Check if it's a part-level node
        if idx >= part_level_start_index:
            is_leaf_in_preds = True
            # Check if any predicted node is a child of this node
            children_indices = np.where(A_unified[idx, :] == 1)[0]
            for child_idx in children_indices:
                 # Although parts shouldn't have children in A_unified, this is safer if structure changes
                if child_idx in pred_indices:
                    is_leaf_in_preds = False
                    break
            if is_leaf_in_preds:
                 valid_leaves.append(idx)

    if valid_leaves:
        # If valid part leaves predicted, choose the one with highest probability
        deepest_node_idx = max(valid_leaves, key=lambda leaf_idx: probabilities[leaf_idx].item())
    elif pred_indices:
        # If no valid part leaves, find deepest among *all* predictions
        # This handles cases where prediction stops at a higher level but is still above threshold
        deepest_node_idx = max(pred_indices, key=lambda idx: len(get_ancestry_path(A_unified, idx)))
    else:
        return [] # No predictions above threshold

    if deepest_node_idx != -1:
        # Return the full ancestry path from the chosen deepest node
        return get_ancestry_path(A_unified, deepest_node_idx)
    else:
        return []


###########################################
# 4. Model Definition (Keep EXACTLY as used for training)
###########################################
# --- Copy the HierarchicalAttentionModel class definition here ---
class HierarchicalAttentionModel(nn.Module):
    def __init__(self, target_dim, hierarchy_levels=5):
        super(HierarchicalAttentionModel, self).__init__()
        # Ensure weights= parameter is correct for your torchvision version
        # Older versions might use pretrained=True
        try:
            self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        except TypeError:
            print("Warning: Using legacy 'pretrained=True'. Update torchvision for 'weights' parameter.")
            self.backbone = models.efficientnet_b0(pretrained=True)

        num_ftrs = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        self.attention = nn.Sequential(
            nn.Linear(num_ftrs, num_ftrs // 2), nn.ReLU(),
            nn.Linear(num_ftrs // 2, num_ftrs), nn.Sigmoid()
        )
        self.fc_shared = nn.Linear(num_ftrs, 512)
        self.bn_shared = nn.BatchNorm1d(512)
        self.level_embeddings = nn.Embedding(hierarchy_levels, 64)
        self.level_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512 + 64, 256), nn.ReLU(),
                nn.Dropout(0.2 + i * 0.05),
                nn.Linear(256, target_dim)
            ) for i in range(hierarchy_levels)
        ])
        # No need to call _init_weights() when loading state_dict

    # No need for _init_weights when loading a trained model

    def forward(self, x):
        features = self.backbone(x)
        att_weights = self.attention(features)
        weighted_features = features * att_weights
        shared_feats = F.relu(self.bn_shared(self.fc_shared(weighted_features)))
        outputs = []
        batch_size = x.size(0)
        for i, level_layer in enumerate(self.level_layers):
            level_idx = torch.full((batch_size,), i, dtype=torch.long, device=x.device)
            level_embedding = self.level_embeddings(level_idx)
            level_input = torch.cat([shared_feats, level_embedding], dim=1)
            level_output = level_layer(level_input)
            outputs.append(level_output)
        combined_output = sum(outputs) / len(outputs)
        return combined_output

###########################################
# 5. Define Transforms (Keep test_transform EXACTLY as used for training)
###########################################
# We need the transform used for the *test* set during evaluation
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# We also need the original train_transform to recreate the split correctly initially
# --- Copy the train_transform definition here ---
# Note: We won't actually *use* the augmentations from train_transform for prediction,
# but the dataset object needs it initially if it was used for the random_split.
from PIL import ImageFilter, ImageOps # Need these for augmentations
class GaussianBlur:
    def __init__(self, radius_min=0.1, radius_max=2.0): self.radius_min, self.radius_max = radius_min, radius_max
    def __call__(self, img): return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(self.radius_min, self.radius_max)))
# Need ImageEnhance if used in RandomColorJitter


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


###########################################
# 6. Dataset Class (Keep full class definition for recreating split)
###########################################
# --- Copy the HierarchicalLegoDataset class definition here ---
class HierarchicalLegoDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = [] # Stores (group, supercls, cls, subcls, part) tuples

        print(f"Scanning dataset directory: {root_dir}")
        part_ids_in_map = set(subclass_mapping.keys())
        valid_image_count = 0
        processed_part_count = 0

        try:
            all_part_dirs = [item for item in os.scandir(root_dir) if item.is_dir() and item.name in part_ids_in_map]
            # Wrap with tqdm for progress bar
            for item in tqdm(all_part_dirs, desc="Scanning Part Folders"):
                part = item.name
                folder = item.path
                processed_part_count += 1
                try:
                    # Retrieve hierarchy once per part
                    subcls = subclass_mapping[part]
                    cls_label = class_mapping.get(subcls)
                    if cls_label is None: continue
                    supercls = superclass_mapping.get(cls_label)
                    if supercls is None: continue
                    group = group_mapping.get(supercls)
                    if group is None: continue
                    label_tuple = (group, supercls, cls_label, subcls, part)

                    # Scan for images within this part's folder
                    for sub_item in os.scandir(folder):
                        if sub_item.is_file() and sub_item.name.lower().endswith((".jpg", ".png", ".jpeg")):
                            self.image_paths.append(sub_item.path)
                            self.labels.append(label_tuple)
                            valid_image_count += 1
                except KeyError as e:
                    print(f"Warning: Hierarchy mapping error for part {part}: {e}")
                    continue
                except OSError as e:
                    print(f"Warning: Could not scan folder {folder}: {e}")
                    continue
        except FileNotFoundError:
             print(f"Error: Dataset directory not found at {root_dir}")
             self.image_paths, self.labels = [], []
        except Exception as e:
             print(f"An unexpected error occurred during dataset scanning: {e}")
             self.image_paths, self.labels = [], []

        print(f"Found {valid_image_count} valid images across {processed_part_count} processed part folders.")
        if not self.image_paths:
             print("Warning: No images were loaded. Check dataset path and structure.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        label_tuple = self.labels[index]

        try:
            img = Image.open(img_path).convert("RGB")
            current_transform = self.transform
            if current_transform:
                img = current_transform(img)

            # Use the ADDED get_target_vector function
            target = get_target_vector(*label_tuple, all_nodes, node_to_idx, A_unified)
            target = torch.tensor(target)
            return img, target

        except Exception as e:
            print(f"Error loading image {img_path}: {e}. Returning placeholder.")
            placeholder_img = torch.zeros(3, 224, 224)
            placeholder_target = torch.zeros(len(all_nodes))
            return placeholder_img, placeholder_target


###########################################
# 7. Prediction Function (Adapted)
###########################################
def predict_random_test_sample(model, dataset, test_indices, idx_in_test, device,
                               all_nodes, node_to_idx, A_unified, threshold=0.5):
    """Loads a specific test sample, predicts, and compares paths."""

    original_idx = test_indices[idx_in_test] # Get the index in the original full dataset

    print(f"\n--- Predicting for Test Sample {idx_in_test+1} (Original Index: {original_idx}) ---")

    # --- IMPORTANT: Apply the TEST transform ---
    original_transform = dataset.transform
    dataset.transform = test_transform # Temporarily set test transform
    try:
        # Get data using original index - this now calls the fixed __getitem__
        image_tensor, target_vector = dataset[original_idx]
        # Handle potential errors during loading inside __getitem__
        if torch.equal(image_tensor, torch.zeros(3, 224, 224)):
             print("Placeholder tensor returned by dataset, skipping prediction for this sample.")
             dataset.transform = original_transform # Restore transform
             return
    except Exception as e:
         print(f"Error getting item {original_idx} from dataset: {e}")
         dataset.transform = original_transform # Restore transform
         return
    finally:
        dataset.transform = original_transform # Restore the original transform

    image_path = dataset.image_paths[original_idx] # Get path for reference
    print(f"Image Path: {image_path}")

    # Prepare tensor for model
    input_tensor = image_tensor.unsqueeze(0).to(device) # Add batch dim and move to device

    # Make prediction
    model.eval() # Ensure model is in eval mode
    with torch.no_grad():
        output_logits = model(input_tensor)

    # Get predicted path indices from logits
    predicted_path_indices = get_predicted_path(output_logits[0], all_nodes, node_to_idx, A_unified, threshold)
    predicted_path_names = [all_nodes[idx] for idx in predicted_path_indices]

    # Get actual path indices from ground truth target vector
    actual_path_indices = torch.where(target_vector > 0.5)[0].numpy().tolist() # Use numpy for consistency
    actual_path_names = [all_nodes[idx] for idx in actual_path_indices]

    # Print comparison
    print(f"Actual Path    : {actual_path_names}")
    print(f"Predicted Path : {predicted_path_names}")

    # Optional: Print probabilities for predicted nodes
    if predicted_path_indices:
         print("Predicted Node Probabilities:")
         probabilities = torch.sigmoid(output_logits[0]).cpu() # Calculate probabilities for display
         for idx in predicted_path_indices:
              print(f"  - {all_nodes[idx]} ({get_node_level(idx, groups, superclasses, classes, subclasses, parts)}): {probabilities[idx]:.4f}")


###########################################
# 8. Main Execution for Prediction
###########################################
if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Set Seed for Reproducible Split ---
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    # --- Load Full Dataset and Recreate Split ---
    print(f"\nLoading full dataset from: {DATA_DIR}")
    # Instantiate with train_transform initially IF it was used for the original split
    full_dataset = HierarchicalLegoDataset(DATA_DIR, transform=train_transform)

    if len(full_dataset) == 0:
        print("Dataset is empty. Cannot proceed.")
        exit()

    print("Recreating train/val/test split...")
    dataset_size = len(full_dataset)
    train_size = int(0.7 * dataset_size)
    val_size = int(0.15 * dataset_size)
    test_size = dataset_size - train_size - val_size

    if dataset_size < 3 or test_size <= 0 :
         print(f"Error: Dataset size ({dataset_size}) too small or calculated test set size ({test_size}) is non-positive. Cannot select test samples.")
         exit()

    # Perform the split using the SAME seed
    train_dataset_subset, val_dataset_subset, test_dataset_subset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(SEED)
    )
    print(f"Split recreated: Train={len(train_dataset_subset)}, Val={len(val_dataset_subset)}, Test={len(test_dataset_subset)}")

    # --- Select Random Test Samples ---
    test_indices = test_dataset_subset.indices # Get the list of original indices in the test set

    if len(test_indices) < NUM_SAMPLES:
        print(f"Warning: Requested {NUM_SAMPLES} samples, but test set only has {len(test_indices)}. Using all test samples.")
        num_samples_to_select = len(test_indices)
    else:
        num_samples_to_select = NUM_SAMPLES

    if num_samples_to_select == 0:
        print("Test set is empty. No samples to predict.")
        exit()

    random_test_indices_in_subset = random.sample(range(len(test_indices)), num_samples_to_select)
    print(f"\nSelected {num_samples_to_select} random indices from the test set.")


    # --- Load Model ---
    print(f"Loading model from: {MODEL_PATH}")
    try:
        model = HierarchicalAttentionModel(target_dim=n_total, hierarchy_levels=5)
        # Load using weights_only=True for security unless you absolutely trust the source
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
        model.to(device)
        model.eval() # Set to evaluation mode
        print("Model loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Model file not found at {MODEL_PATH}. Cannot perform predictions.")
        exit()
    except Exception as e:
        print(f"Error loading model: {e}")
        exit()

    # --- Predict for each selected random sample ---
    print("\nStarting predictions on random test samples...")
    for i in random_test_indices_in_subset:
        # The index 'i' here is the index *within the test_dataset subset*
        # We need the corresponding index in the original full_dataset, which is test_indices[i]
        predict_random_test_sample(
            model=model,
            dataset=full_dataset, # Pass the original dataset
            test_indices=test_indices, # Pass the list of test indices
            idx_in_test=i,         # Pass the index within the test subset
            device=device,
            all_nodes=all_nodes,
            node_to_idx=node_to_idx,
            A_unified=A_unified,
            threshold=PREDICTION_THRESHOLD
        )

    print("\n--- Prediction complete ---")