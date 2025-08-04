#!/bin/bash

# Set target directory
TARGET_DIR="./coco"
mkdir -p "$TARGET_DIR"

# Navigate to target directory
cd "$TARGET_DIR" || exit

echo "Downloading COCO 2017 images and annotations..."

# Image zip URLs
IMG_URLS=(
  "http://images.cocodataset.org/zips/train2017.zip"
  "http://images.cocodataset.org/zips/val2017.zip"
  "http://images.cocodataset.org/zips/test2017.zip"
)

# Annotation zip URLs
ANN_URLS=(
  "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
  "http://images.cocodataset.org/annotations/image_info_test2017.zip"
)

# Download and unzip image zips
for url in "${IMG_URLS[@]}"; do
  fname=$(basename "$url")
  echo "Downloading $fname..."
  curl -LO "$url"
  echo "Unzipping $fname..."
  unzip -q "$fname"
  rm "$fname"
done

# Download and unzip annotation zips
for url in "${ANN_URLS[@]}"; do
  fname=$(basename "$url")
  echo "Downloading $fname..."
  curl -LO "$url"
  echo "Unzipping $fname..."
  unzip -q "$fname"
  rm "$fname"
done

echo "COCO 2017 data is ready in $TARGET_DIR"