import os
import numpy as np
from collections import defaultdict

GT_TXT_FOLDER   = "./dataset/surfing/groundTruth"
OUTPUT_BOUNDARY = "./dataset/surfing/boundary"
MAPPING_SAVE    = "./dataset/surfing/mapping.txt"

if not os.path.exists(OUTPUT_BOUNDARY):
    os.makedirs(OUTPUT_BOUNDARY)
action_set = set()
print("Scanning all action labels...")

for txt_name in os.listdir(GT_TXT_FOLDER):
    if not txt_name.endswith(".txt"):
        continue
    try:
        with open(os.path.join(GT_TXT_FOLDER, txt_name), "r") as f:
            words = [line.strip() for line in f if line.strip()]
            action_set.update(words)
    except:
        continue

action_list = sorted(list(action_set))
action2id = {act: i for i, act in enumerate(action_list)}

print("Detected action classes:")
for act, idx in action2id.items():
    print(f"   {idx} <- {act}")

with open(MAPPING_SAVE, "w") as f:
    for act, idx in action2id.items():
        f.write(f"{act} {idx}\n")

print(f"\nMapping file saved to: {MAPPING_SAVE}")

def generate_boundary(label_list):
    length = len(label_list)
    boundary = np.zeros(length, dtype=np.float32)
    if length == 0:
        return boundary
    boundary[0] = 1.0
    for i in range(1, length):
        if label_list[i] != label_list[i-1]:
            boundary[i] = 1.0
    return boundary

print("\nGenerating boundary files...")
txt_files = [f for f in os.listdir(GT_TXT_FOLDER) if f.endswith(".txt")]

for idx, txt_name in enumerate(txt_files):
    base_name = txt_name[:-4]
    txt_path = os.path.join(GT_TXT_FOLDER, txt_name)

    try:
        with open(txt_path, "r") as f:
            words = [line.strip() for line in f if line.strip()]
            labels = [action2id[w] for w in words]
        boundary = generate_boundary(labels)
        save_path = os.path.join(OUTPUT_BOUNDARY, f"{base_name}.npy")
        np.save(save_path, boundary)

        print(f"[{idx+1}/{len(txt_files)}] Processed: {base_name}.npy  Frames:{len(labels)}")

    except Exception as e:
        print(f"[{idx+1}/{len(txt_files)}] Failed: {txt_name} | {str(e)}")

print("\nAll tasks completed!")
