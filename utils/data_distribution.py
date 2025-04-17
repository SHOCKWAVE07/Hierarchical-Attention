import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm # Optional: for progress bar during dataset loading

# --- Keep only necessary imports ---

###########################################
# 1. Define the hierarchical mappings
###########################################

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

# ===============================
# 2. Create Node Sets for Each Hierarchical Level
# ===============================
groups = sorted(set(group_mapping.values()))
superclasses = sorted(set(group_mapping.keys()))
classes = sorted(set(class_mapping.values()))
subclasses = sorted(set(subclass_mapping.values()))
parts = sorted(set(subclass_mapping.keys()))

print(f"Found {len(groups)} Groups, {len(superclasses)} Superclasses, {len(classes)} Classes, {len(subclasses)} Subclasses, {len(parts)} Parts.")

# ===============================
# 3. Dataset Class (Simplified for this purpose)
# ===============================
class LegoLabelDataset(Dataset):
    """Loads only the hierarchical labels, not images."""
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.labels = [] # List to store (group, supercls, cls, subcls, part) tuples
        self.valid_image_count = 0

        print("Scanning dataset directory for labels...")
        # Process each potential part directory
        part_folders = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

        for part in tqdm(sorted(part_folders), desc="Scanning Parts"):
            # Check if this folder name is a valid Part ID in our hierarchy
            if part not in subclass_mapping:
                # print(f"Skipping folder '{part}' - not found in subclass_mapping.")
                continue

            folder = os.path.join(root_dir, part)
            has_valid_image = False
            for fname in os.listdir(folder):
                if fname.lower().endswith((".jpg", ".png", ".jpeg")):
                    # We just need to know there's at least one image for this part
                    # No need to open it for label counting
                    has_valid_image = True
                    self.valid_image_count += 1
                    break # Only need one image per part folder to count the labels

            if has_valid_image:
                # If a valid image exists, retrieve the full hierarchy for this part
                try:
                    subcls = subclass_mapping[part]
                    cls_label = class_mapping.get(subcls)
                    if cls_label is None: continue
                    supercls = superclass_mapping.get(cls_label)
                    if supercls is None: continue
                    group = group_mapping.get(supercls)
                    if group is None: continue

                    # Count one label tuple for each valid image found in this part's folder
                    # Re-scan to count all images for accurate distribution
                    image_count_for_part = 0
                    for fname in os.listdir(folder):
                         if fname.lower().endswith((".jpg", ".png", ".jpeg")):
                              image_count_for_part += 1

                    # Add the label tuple multiple times, once for each image
                    for _ in range(image_count_for_part):
                        self.labels.append((group, supercls, cls_label, subcls, part))

                except KeyError as e:
                    print(f"Hierarchy mapping error for part {part}: {e}")
                    continue

        print(f"Found {self.valid_image_count} image files corresponding to {len(self.labels)} label entries across {len(set(p for _,_,_,_,p in self.labels))} unique parts.")
        if not self.labels:
            print("Warning: No labels were loaded. Check dataset path and hierarchy mappings.")


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # We only need the labels for distribution plotting
        return self.labels[index]


###########################################
# 4. Function to Plot Data Distribution for All Levels
###########################################

def plot_data_distribution_all_levels(dataset_labels, output_dir, groups, superclasses, classes, subclasses, parts):
    """
    Plots and saves the distribution of samples across categories
    for each level of the hierarchy based on the provided label list.

    Args:
        dataset_labels (list): A list of label tuples (group, supercls, cls, subcls, part).
        output_dir (str): Directory to save the plots.
        groups (list): List of Group names.
        superclasses (list): List of Superclass names.
        classes (list): List of Class names.
        subclasses (list): List of Subclass names.
        parts (list): List of Part names.
    """
    print("\nGenerating data distribution plots for all levels...")

    if not dataset_labels:
        print("Label list is empty. Cannot generate distribution plots.")
        return

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Define mapping from level name to index in the label tuple
    level_info = {
        "Group": {"index": 0},
        "Superclass": {"index": 1},
        "Class": {"index": 2},
        "Subclass": {"index": 3},
        "Part": {"index": 4}
    }

    # Iterate through each level
    for level_name, info in level_info.items():
        level_index = info["index"]
        print(f"  Processing level: {level_name}")

        # Count occurrences of each category at this level
        counts = Counter()
        for label_tuple in dataset_labels:
            try:
                category_name = label_tuple[level_index]
                counts[category_name] += 1
            except IndexError:
                print(f"Warning: Label tuple {label_tuple} seems malformed. Skipping.")
                continue

        if not counts:
            print(f"  No data found for level {level_name}. Skipping plot.")
            continue

        # Prepare data for plotting (sort alphabetically by category name)
        sorted_items = sorted(counts.items())
        labels, values = zip(*sorted_items)

        # Create the plot
        plt.figure(figsize=(max(15, len(labels) * 0.25), 9)) # Dynamic width, adjust multiplier as needed
        bars = plt.bar(labels, values, color='cornflowerblue') # Changed color
        plt.xlabel(level_name, fontsize=12)
        plt.ylabel('Number of Samples', fontsize=12)
        plt.title(f'Data Distribution by {level_name}', fontsize=14, fontweight='bold')
        plt.xticks(rotation=90, fontsize=8) # Rotate labels for readability
        plt.yticks(fontsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Add counts on top of bars if there aren't too many bars
        if len(labels) <= 50: # Adjust this threshold as needed
             for bar in bars:
                 yval = bar.get_height()
                 plt.text(bar.get_x() + bar.get_width()/2.0, yval, int(yval), va='bottom', ha='center', fontsize=7) # Add text labels

        plt.tight_layout() # Adjust layout to prevent labels overlapping

        # Save the plot
        save_path = os.path.join(output_dir, f'data_distribution_{level_name.lower()}.png')
        try:
            plt.savefig(save_path, dpi=150) # Increase DPI for better resolution
            print(f"  Distribution plot saved to {save_path}")
        except Exception as e:
            print(f"  Error saving plot for {level_name}: {e}")

        plt.close() # Close the figure to free memory

    print("Finished generating distribution plots.")


###########################################
# 5. Main Script Execution
###########################################
def main_plot_distribution():
    """Loads dataset labels and generates distribution plots."""

    # --- Configuration ---
    # !!! IMPORTANT: Set your data directory path here !!!
    data_dir = "/scratch/m24csa026/DL-Project/64"
    # Set the directory where plots will be saved
    output_dir = "output_distribution_plots/"
    # --- End Configuration ---

    print("-" * 50)
    print("Starting Data Distribution Analysis")
    print(f"Dataset directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print("-" * 50)

    # 1. Load only the labels from the dataset
    # No transforms needed for just counting labels
    label_dataset = LegoLabelDataset(root_dir=data_dir)

    # Check if any labels were loaded
    if len(label_dataset) == 0:
        print("\nNo data loaded. Exiting.")
        return

    # 2. Generate and save plots for all levels
    plot_data_distribution_all_levels(
        dataset_labels=label_dataset.labels, # Pass the list of label tuples
        output_dir=output_dir,
        groups=groups,
        superclasses=superclasses,
        classes=classes,
        subclasses=subclasses,
        parts=parts
    )

    print("\nData distribution analysis complete.")
    print(f"Plots saved in: {output_dir}")
    print("-" * 50)

if __name__ == "__main__":
    # Ensure all necessary global variables (mappings, node lists)
    # are defined before calling the main function.
    main_plot_distribution()