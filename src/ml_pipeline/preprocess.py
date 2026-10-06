import os
import splitfolders

def clean_and_split_data():
    base_dir = os.getcwd()
    raw_base = os.path.join(base_dir, "data", "raw")
    processed_data_path = os.path.join(base_dir, "data", "processed")
    
    # Check if PlantVillage or inner PlantVillage exists
    raw_data_path = os.path.join(raw_base, "PlantVillage")
    
    # If the unzipped folder created a nested PlantVillage/PlantVillage or plantvillage directory
    if not os.path.exists(raw_data_path):
        # Look for any directory inside data/raw
        subdirs = [os.path.join(raw_base, d) for d in os.listdir(raw_base) if os.path.isdir(os.path.join(raw_base, d))]
        if subdirs:
            raw_data_path = subdirs[0]

    print(f"📂 Reading raw data from: {raw_data_path}")
    
    # Automatically performs the split (80/10/10)
    splitfolders.ratio(
        input=raw_data_path,
        output=processed_data_path,
        seed=42,
        ratio=(0.8, 0.1, 0.1),
        move=False
    )
    
    print("\n✅ Success! Clean dataset split created.")

if __name__ == "__main__":
    clean_and_split_data()