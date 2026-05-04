import os
import numpy as np

def flatten_npy_features(folder_path, file_suffix=".npy"):
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Error: Path does not exist")
    
    npy_files = [
        filename for filename in os.listdir(folder_path)
        if filename.lower().endswith(file_suffix.lower())
    ]
    
    if len(npy_files) == 0:
        print(f"Warning: No {file_suffix} files found in {folder_path}")
        return
    
    print(f"Starting processing: Found {len(npy_files)} .npy feature files")
    
    for filename in npy_files:
        file_path = os.path.join(folder_path, filename)
        try:
            features_3d = np.load(file_path)
        except Exception as e:
            print(f"Skipped: Failed to load {filename} -> {e}")
            continue
        
        if len(features_3d.shape) != 3 or features_3d.shape[1:] != (17, 2):
            print(f"Skipped: {filename} has invalid shape {features_3d.shape}, expected (n_frames, 17, 2)")
            continue
        
        features_2d = features_3d.reshape(features_3d.shape[0], -1)
        np.save(file_path, features_2d)
        print(f"Processed: {filename} -> shape changed from {features_3d.shape} to {features_2d.shape} (34-dim features)")
    
    print("Batch feature flattening completed!")

if __name__ == "__main__":
    target_folder = "/home/zzy/qml/nips/data/features"
    try:
        flatten_npy_features(folder_path=target_folder)
    except Exception as e:
        print(f"Program error: {e}")

