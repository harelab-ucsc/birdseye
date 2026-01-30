import os
import sys
import shutil

if len(sys.argv) != 4:
    print(f"Usage: {sys.argv[0]} <directory> <N> <destination>")
    sys.exit(1)

dir_path = sys.argv[1]
try:
    step = int(sys.argv[2])
except ValueError:
    print("N must be an integer.")
    sys.exit(1)

dest_path = sys.argv[3]

if not os.path.isdir(dir_path):
    print(f"Error: {dir_path} is not a directory.")
    sys.exit(1)

if not os.path.exists(dest_path):
    os.makedirs(dest_path)

# collect jpg files (case-insensitive)
files = [f for f in os.listdir(dir_path) if f.lower().endswith(".jpg")]

# sort by filename
files.sort()

# take every N-th file
selected = files[::step]

# copy files to destination
for f in selected:
    src = os.path.join(dir_path, f)
    dst = os.path.join(dest_path, f)
    shutil.copy2(src, dst)
    print(f"Copied: {src} -> {dst}")
