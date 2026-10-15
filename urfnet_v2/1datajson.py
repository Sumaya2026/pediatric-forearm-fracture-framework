import json
from pathlib import Path

root = Path(r"/autodl-fs/data/dataset/test")
json_path = Path(r"/root/autodl-tmp/side/test.json")

img_list = []

for p in root.rglob("*.jpg"):
    if p.is_file() and (p.name == "front.jpg" or p.name == "side.jpg"):
        img_list.append(str(p.resolve()))

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(img_list, f, ensure_ascii=False, indent=2)








