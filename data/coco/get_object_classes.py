from pycocotools.coco import COCO

# Load COCO annotations
coco = COCO("./annotations/instances_train2017.json")

# Get category info
cats = coco.loadCats(coco.getCatIds())
category_names = [cat['name'] for cat in cats]

# Write to file
with open("./coco_classes.txt", "w") as f:
    for name in category_names:
        f.write(name + "\n")