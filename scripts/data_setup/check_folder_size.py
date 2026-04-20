import os
from pathlib import Path

def get_folder_size(path):
    total = 0
    file_count = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            try:
                total += os.path.getsize(filepath)
                file_count += 1
            except Exception:
                pass
    return total, file_count

print("=== Data Folder Structure ===\n")

# Check chunks_clean
chunks_clean_path = 'data/chunks_clean'
if os.path.exists(chunks_clean_path):
    size, count = get_folder_size(chunks_clean_path)
    print(f"data/chunks_clean/")
    print(f"  Total size: {size / 1e9:.2f} GB")
    print(f"  File count: {count}")
    
    # List top-level items
    print(f"\n  Top-level contents:")
    for item in os.listdir(chunks_clean_path):
        item_path = os.path.join(chunks_clean_path, item)
        if os.path.isfile(item_path):
            size_mb = os.path.getsize(item_path) / 1e6
            print(f"    - {item} ({size_mb:.2f} MB) [file]")
        elif os.path.isdir(item_path):
            item_size, item_count = get_folder_size(item_path)
            size_mb = item_size / 1e6
            print(f"    - {item}/ ({size_mb:.2f} MB, {item_count} files) [directory]")
else:
    print("ERROR: data/chunks_clean not found\n")

# Check entire data folder
print("\n=== Total Data Folder ===\n")
data_path = 'data'
size, count = get_folder_size(data_path)
print(f"data/ (all subfolders)")
print(f"  Total size: {size / 1e9:.2f} GB")
print(f"  Total files: {count}")

# Breakdown by subfolder
print(f"\n  Breakdown by subfolder:")
for subfolder in os.listdir(data_path):
    subfolder_path = os.path.join(data_path, subfolder)
    if os.path.isdir(subfolder_path):
        sub_size, sub_count = get_folder_size(subfolder_path)
        size_gb = sub_size / 1e9
        print(f"    - {subfolder}/: {size_gb:.2f} GB ({sub_count} files)")

print("\n=== Upload Assessment ===\n")
data_size_gb = size / 1e9
print(f"Total data folder size: {data_size_gb:.2f} GB")
print(f"\nGoogle Drive storage:")
print(f"  Free tier: 15 GB available")
print(f"  Can upload: {'YES ✓' if data_size_gb < 15 else 'NO - requires paid storage'}")
print(f"\nRecommendation:")
if data_size_gb < 5:
    print(f"  ✓ Upload entire data/ folder to Drive")
elif data_size_gb < 15:
    print(f"  ✓ Upload data/chunks_clean/ subfolder (this is what Colab needs)")
else:
    print(f"  ! Upload chunks_clean/ only; keep raw/ and processing files local")
