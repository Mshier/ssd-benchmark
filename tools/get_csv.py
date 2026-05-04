import os

LIST_DIR = "./datasets/surfing/"
CSV_OUTPUT = "./csv/surfing/"

os.makedirs(CSV_OUTPUT, exist_ok=True)

files = [
    ("train.list", "train1.csv"),
    ("val.list", "val1.csv"),
    ("test.list", "test1.csv"),
]

def generate_csv(list_name, csv_name):
    list_path = os.path.join(LIST_DIR, list_name)
    csv_path = os.path.join(CSV_OUTPUT, csv_name)

    with open(list_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("feature,label,boundary\n")
        for line in lines:
            name = line.rsplit(".", 1)[0]
            csv_line = f"/surfing/features/{name}.npy,/surfing/groundTruth/{name}.npy,/surfing/boundary/{name}.npy"
            f.write(csv_line + "\n")

for list_name, csv_name in files:
    if os.path.exists(os.path.join(LIST_DIR, list_name)):
        generate_csv(list_name, csv_name)

