import os
from collections import Counter

label_dir = "datasets/kitti/label_2"
counter = Counter()

for file in os.listdir(label_dir):
    with open(os.path.join(label_dir, file)) as f:
        for line in f:
            cls = line.split()[0]
            counter[cls] += 1

print(counter)
