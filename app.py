import streamlit as st
import torch
import numpy as np
from PIL import Image
from torchvision import transforms, models
import time

# Define all necessary mappings and hierarchy structure (same as training)
# [PASTE ALL MAPPING DICTIONARIES (subclass_mapping, class_mapping, etc.) HERE]
# [PASTE THE get_node_level FUNCTION HERE]
# [PASTE THE get_ancestry_path FUNCTION HERE]
# [PASTE THE NODE STRUCTURE CREATION CODE (groups, superclasses, etc.) HERE]
# [PASTE THE A_unified MATRIX CREATION CODE HERE]

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

# Define the model architecture (must match training)
class HierarchicalAttentionModel(torch.nn.Module):
    def __init__(self, target_dim, hierarchy_levels=5):
        super().__init__()
        # Use EXACTLY the same backbone as in training
        self.backbone = models.efficientnet_b0(pretrained=False)
        num_ftrs = self.backbone.classifier[1].in_features
        self.backbone.classifier = torch.nn.Identity()
        
        # Maintain identical architecture
        self.attention = torch.nn.Sequential(
            torch.nn.Linear(num_ftrs, num_ftrs // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(num_ftrs // 2, num_ftrs),
            torch.nn.Sigmoid()
        )
        
        self.fc_shared = torch.nn.Linear(num_ftrs, 512)
        self.bn_shared = torch.nn.BatchNorm1d(512)
        
        self.level_embeddings = torch.nn.Embedding(hierarchy_levels, 64)
        
        self.level_layers = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(512 + 64, 256),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.2 + i * 0.05),
                torch.nn.Linear(256, target_dim)
            ) for i in range(hierarchy_levels)
        ])

    def forward(self, x):
        features = self.backbone(x)
        att_weights = self.attention(features)
        weighted_features = features * att_weights
        shared_feats = self.fc_shared(weighted_features)
        shared_feats = self.bn_shared(shared_feats)
        shared_feats = torch.nn.functional.relu(shared_feats)
        
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

@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HierarchicalAttentionModel(len(all_nodes)).to(device)
    
    # Load state dict with strict mapping
    state_dict = torch.load('best_model.pth', map_location=device)
    
    # Handle any potential DataParallel wrapping
    if next(iter(state_dict)).startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, device


# Image preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# Streamlit app
def main():
    st.title("LEGO Part Hierarchy Classifier")
    st.write("Upload an image of a LEGO part to see its hierarchical classification")

    # Load model
    model, device = load_model()

    # File upload
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display image
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', width=300)

        # Preprocess and predict
        with st.spinner('Analyzing...'):
            start_time = time.time()
            
            # Preprocess
            input_tensor = preprocess_image(image).to(device)
            
            # Predict
            with torch.no_grad():
                output = model(input_tensor)
                probs = torch.sigmoid(output)[0]
            
            # Get prediction path
            threshold = 0.5
            pred_indices = torch.where(probs > threshold)[0].cpu().numpy().tolist()
            if pred_indices:
                leaf = max(pred_indices, key=lambda idx: len(get_ancestry_path(A_unified, idx)))
                pred_path = get_ancestry_path(A_unified, leaf)
            else:
                pred_path = []
            
            processing_time = time.time() - start_time

        # Display results
        st.success(f"Analysis complete! Processing time: {processing_time:.2f}s")
        
        if not pred_path:
            st.error("No valid hierarchy found. Please try another image.")
            return

        # Organize predictions by level
        levels = {}
        for idx in pred_path:
            node_name = all_nodes[idx]
            level = get_node_level(idx, groups, superclasses, classes, subclasses, parts)
            levels.setdefault(level, []).append((node_name, probs[idx].item()))
        
        # Display hierarchy with confidence scores
        st.subheader("Predicted Hierarchy:")
        level_order = ['Group', 'Superclass', 'Class', 'Subclass', 'Part']
        
        for level in level_order:
            if level in levels:
                with st.expander(f"{level} Details", expanded=True):
                    for node_name, confidence in levels[level]:
                        st.write(f"{node_name} ({confidence*100:.1f}%)")
                        st.progress(confidence)

if __name__ == "__main__":
    main()