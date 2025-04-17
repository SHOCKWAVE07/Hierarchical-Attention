# lego_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np

subclass_mapping = {
    "3001": "2x4 Brick","3002": "2x3 Brick","3003": "2x2 Brick","2357": "Corner Brick","3622": "Arch Brick",
    "3009": "Sloped Brick","3004": "2x6 Brick","3005": "2x8 Brick","3008": "2x10 Brick","3020": "2x4 Plate",
    "3021": "2x3 Plate","3023": "2x2 Plate","3832": "Baseplate","3665": "Grille Plate","3460": "Sloped Plate",
    "3024": "1x1 Round Plate","4477": "1x10 Plate","60479": "1x12 Plate","2431": "2x2 Corner Tile","6636": "1x6 Tile",
    "4162": "1x8 Tile","63864": "Tile with Hole","2436": "Grooved Tile","3068b": "Inverted Tile","32278": "Slope Tile", # key 32278
    "22388": "Curved Tile","32523": "Rotation Tile", # key 32523 - Using Rotation Tile over Rotation Joint
    "48729": "Bar Holder","60474": "Clip with Handle","15712": "Double Clip","63868": "1x2 Clip","60478": "Vertical Handle",
    "60470": "Pin Clip","88072": "Angled Clip","99207": "Bracket with Holes","99780": "Robot Arm","2429": "Hinge Plate",
    "2430": "Hinge Brick","3937": "1x2 Hinge","3705": "3L Axle","6536": "8L Axle","60451": "16L Axle",
    "3673": "Turntable",
    # "32523": "Rotation Joint", # Duplicate key 32523
    "32524": "Axle Connector", # key 32524 - Using Axle Connector over 5x5 Panel
    # "32524": "5x5 Panel", # Duplicate key 32524
    "32525": "Curved Panel","32526": "Sloped Panel","53451": "24-Tooth Gear","54200": "Bevel Gear","60483": "Worm Gear",
    "48336": "Axle Connector", # This is a different part number mapping to same subclass name
    "54201": "Axle with Stop","60484": "Axle with Pin","15573": "Jumper Plate","18651": "Paint Roller",
    "60471": "Pin Clip Tool","2586": "Drumstick","79743": "Cupcake","30229": "Zipline Handle","27150": "Blade",
    "27167": "Treasure Chest Lid","30230": "Weapon Handle","6854": "Fern",
    "50950": "Plant Wedge", # key 50950 - Using Plant Wedge over 10 Degree Wedge
    "3565": "Plant Leaf","77822": "Ramp Loop","35646": "Rock","1998": "Terrain Piece","25893": "Horse Hitching",
    "11203": "Trap Door","50951": "Animal Wedge","3623": "1x3 Arch","3224": "2x2 Arch","32279": "Curved Slope",
    "22885": "1x2x3 Cylinder","43888": "1x1x6 Round","22389": "Curved Cylinder",
    # "32278": "45° Wedge", # Duplicate key 32278
    "22390": "Curved Wedge",
    # "50950": "10° Wedge" # Duplicate key 50950
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
    "Turntable": "Rotation Parts", #"Rotation Joint": "Rotation Parts", # Needs resolution based on subclass_mapping
    "Axle Connector": "Rotation Parts", # Mapping for 'Axle Connector' subclass name
    "5x5 Panel": "Panels", "Curved Panel": "Panels", "Sloped Panel": "Panels",
    "24-Tooth Gear": "Gears", "Bevel Gear": "Gears", "Worm Gear": "Gears",
    "Axle with Stop": "Axle Accessories", "Axle with Pin": "Axle Accessories",
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
groups = sorted(list(set(group_mapping.values())))
superclasses = sorted(list(set(group_mapping.keys())))
classes = sorted(list(set(class_mapping.values())))
unique_subclass_names = sorted(list(set(subclass_mapping.values())))
part_to_subclass_name = {part: name for part, name in subclass_mapping.items()}
parts = sorted(list(set(subclass_mapping.keys())))
all_nodes = groups + superclasses + classes + unique_subclass_names + parts
node_to_idx = {node: idx for idx, node in enumerate(all_nodes)}
idx_to_node = {idx: node for node, idx in node_to_idx.items()}
n_total = len(all_nodes)
A_unified = np.zeros((n_total, n_total), dtype=int)
for sc in superclasses:
    group = group_mapping.get(sc)
    if group and group in node_to_idx and sc in node_to_idx:
        A_unified[node_to_idx[group], node_to_idx[sc]] = 1
for cls, sc in superclass_mapping.items():
    if sc in node_to_idx and cls in node_to_idx:
        A_unified[node_to_idx[sc], node_to_idx[cls]] = 1
for subcls_name, cls in class_mapping.items():
     if subcls_name in unique_subclass_names and cls in node_to_idx:
         if subcls_name in node_to_idx:
             A_unified[node_to_idx[cls], node_to_idx[subcls_name]] = 1
for part, subcls_name in part_to_subclass_name.items():
    if subcls_name in node_to_idx and part in node_to_idx:
        A_unified[node_to_idx[subcls_name], node_to_idx[part]] = 1

# --- Helper Functions --- (Keep get_mappings_and_sizes, get_ancestry_path, get_node_level, get_predicted_path as before)
def get_mappings_and_sizes():
    part_to_idx_map = {part: idx for idx, part in enumerate(parts)}
    idx_to_part_map = {idx: part for part, idx in part_to_idx_map.items()}
    idx_to_part_name_map = {idx: part_to_subclass_name.get(part, "Unknown") for idx, part in idx_to_part_map.items()}
    num_parts_calc = len(part_to_idx_map)
    subclass_to_idx_map = {name: idx for idx, name in enumerate(unique_subclass_names)}
    idx_to_subclass_map = {idx: name for name, idx in subclass_to_idx_map.items()}
    num_subclasses_calc = len(subclass_to_idx_map)
    class_to_idx_map = {name: idx for idx, name in enumerate(classes)}
    idx_to_class_map = {idx: name for name, idx in class_to_idx_map.items()}
    num_classes_calc = len(class_to_idx_map)
    superclass_to_idx_map = {name: idx for idx, name in enumerate(superclasses)}
    idx_to_superclass_map = {idx: name for name, idx in superclass_to_idx_map.items()}
    num_superclasses_calc = len(superclass_to_idx_map)
    group_to_idx_map = {name: idx for idx, name in enumerate(groups)}
    idx_to_group_map = {idx: name for name, idx in group_to_idx_map.items()}
    num_groups_calc = len(group_to_idx_map)
    return (num_parts_calc, num_subclasses_calc, num_classes_calc, num_superclasses_calc, num_groups_calc,
            idx_to_part_map, idx_to_part_name_map, idx_to_subclass_map, idx_to_class_map, idx_to_superclass_map, idx_to_group_map)

def get_ancestry_path(A, leaf_idx):
    if leaf_idx >= A.shape[0] or leaf_idx < 0: return []
    path = [leaf_idx]; current = leaf_idx; visited_in_path = {leaf_idx}
    while True:
        parents = np.where(A[:, current] == 1)[0]
        if len(parents) == 0: break
        parent = parents[0]
        if parent in visited_in_path: break
        path.insert(0, parent); visited_in_path.add(parent); current = parent
    return path

def get_node_level(node_idx, groups_list, superclasses_list, classes_list, subclasses_list, parts_list):
    len_g = len(groups_list); len_sc = len(superclasses_list); len_c = len(classes_list); len_s = len(subclasses_list)
    if 0 <= node_idx < len_g: return "Group"
    elif len_g <= node_idx < len_g + len_sc: return "Superclass"
    elif len_g + len_sc <= node_idx < len_g + len_sc + len_c: return "Class"
    elif len_g + len_sc + len_c <= node_idx < len_g + len_sc + len_c + len_s: return "Subclass"
    elif len_g + len_sc + len_c + len_s <= node_idx < n_total: return "Part"
    else: return "Unknown"

def get_predicted_path(output_probs, all_nodes_list, node_to_idx_map, A_matrix, threshold=0.5):
    if isinstance(output_probs, torch.Tensor): output_probs = output_probs.squeeze().cpu().numpy()
    elif not isinstance(output_probs, np.ndarray): output_probs = np.array(output_probs)
    if output_probs.ndim > 1: output_probs = output_probs.squeeze()
    if output_probs.ndim == 0: return []
    pred_indices = np.where(output_probs > threshold)[0].tolist()
    valid_indices = [idx for idx in pred_indices if idx < A_matrix.shape[0]]
    leaf_level_start_index = len(groups) + len(superclasses) + len(classes) + len(unique_subclass_names)
    if not valid_indices:
        leaf_probs = output_probs[leaf_level_start_index:]
        if len(leaf_probs) > 0:
             highest_prob_leaf_local_idx = np.argmax(leaf_probs)
             highest_prob_leaf_global_idx = highest_prob_leaf_local_idx + leaf_level_start_index
             return get_ancestry_path(A_matrix, highest_prob_leaf_global_idx)
        else: return []
    deepest_node_idx = -1; max_depth = -1
    for idx in valid_indices:
        is_leaf = idx >= leaf_level_start_index; path = get_ancestry_path(A_matrix, idx)
        depth = len(path); effective_depth = depth + (0.5 if is_leaf else 0)
        if effective_depth > max_depth: max_depth = effective_depth; deepest_node_idx = idx
    if deepest_node_idx != -1: return get_ancestry_path(A_matrix, deepest_node_idx)
    else:
        highest_prob_idx = valid_indices[np.argmax(output_probs[valid_indices])]
        return get_ancestry_path(A_matrix, highest_prob_idx)


# --- 4. Model Definition (MATCHING THE SAVED WEIGHTS) ---
class HierarchicalAttentionModel(nn.Module):
    def __init__(self, target_dim, hierarchy_levels=5):
        super(HierarchicalAttentionModel, self).__init__()

        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        num_ftrs = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()

        self.attention = nn.Sequential(
            nn.Linear(num_ftrs, num_ftrs // 2), nn.ReLU(),
            nn.Linear(num_ftrs // 2, num_ftrs), nn.Sigmoid()
        )
        self.fc_shared = nn.Linear(num_ftrs, 512)
        self.bn_shared = nn.BatchNorm1d(512)

        # --- Use the Level-Specific Layers (as indicated by saved keys) ---
        self.level_embeddings = nn.Embedding(hierarchy_levels, 64)
        self.level_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512 + 64, 256),       # Input size combines shared feats + embedding
                nn.ReLU(),
                nn.Dropout(0.2 + i * 0.05),     # Dropout matching training code
                nn.Linear(256, target_dim)      # Output layer for this level
            ) for i in range(hierarchy_levels)
        ])
        # --- Comment out the single output layer ---
        # self.output_layer = nn.Linear(512, target_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding): # Initialize embeddings too
                 nn.init.normal_(m.weight, mean=0, std=0.02)


    def forward(self, x):
        features = self.backbone(x)
        att_weights = self.attention(features)
        weighted_features = features * att_weights
        shared_feats = self.fc_shared(weighted_features)
        shared_feats = self.bn_shared(shared_feats)
        shared_feats = F.relu(shared_feats)

        # --- Use the Level-Specific Forward Logic ---
        outputs = []
        batch_size = x.size(0)
        for i, level_layer in enumerate(self.level_layers):
            # Get level embedding for the batch
            level_idx = torch.full((batch_size,), i, dtype=torch.long, device=x.device)
            level_embedding = self.level_embeddings(level_idx)
            # Concatenate shared features with level embedding
            level_input = torch.cat([shared_feats, level_embedding], dim=1)
            # Pass through the level-specific layer
            level_output = level_layer(level_input)
            outputs.append(level_output)

        # Average the outputs from all levels
        # (This was the logic in your training code's forward pass)
        combined_output = sum(outputs) / len(outputs)
        return combined_output
        # --- End Level-Specific Forward Logic ---

        # --- Comment out single output layer logic ---
        # output_logits = self.output_layer(shared_feats)
        # return output_logits