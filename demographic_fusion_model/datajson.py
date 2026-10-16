import json
from pathlib import Path

root = Path(r"/autodl-fs/data/dataset/train")
json_path = Path(r"/root/autodl-tmp/train.json")

img_list = []

for p in root.rglob("front.jpg"):
    if p.is_file():
        img_list.append(str(p.resolve()))

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(img_list, f, ensure_ascii=False, indent=2)








