#!/usr/bin/env python3
import torch
import os
import pandas as pd
from PIL import Image
import torch.nn.functional as F
import chromadb
from transformers import AutoImageProcessor, AutoModel
import torch

processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
model = AutoModel.from_pretrained('facebook/dinov2-base')
model.eval()
client = chromadb.PersistentClient(path="./db")
collection = client.get_or_create_collection(name="clothes")

def get_embedding(image):
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        vector = outputs.last_hidden_state[:, 0, :].squeeze().tolist()
    return vector

DATA_DIR = "./data"

folder_mapping = {}

for folder_name in os.listdir(DATA_DIR):
    folder_path = os.path.join(DATA_DIR, folder_name)

    if os.path.isdir(folder_path):
        clean_name = folder_name.lstrip('_')
        parts = clean_name.split('_')

        if parts[0].isdigit():
            item_number = parts[0]
            folder_mapping[item_number] = folder_path

print(f"Found {len(folder_mapping)} valid item folders.")

df = pd.read_csv("database.csv")

for index, row in df.iterrows():
    item_number = str(row['Item Number'])
    category = str(row['Category'])

    if item_number not in folder_mapping:
        print(f"Warning: Folder for item {item_number} not found in /data. Skipping.")
        continue

    folder_path = folder_mapping[item_number]

    for img_name in os.listdir(folder_path):
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        path = os.path.join(folder_path, img_name)

        try:
            image = Image.open(path).convert("RGB")
            emb = get_embedding(image)

            img_id = f"{item_number}_{img_name}"

            collection.add(
                ids=[img_id],
                embeddings=[emb],
                metadatas=[{
                    "item_number": item_number,
                    "category": category,
                    "path": path
                }]
            )

            print(f"Added: {path} | Category: {category}")

        except Exception as e:
            print(f"Error on {path}: {e}")

print("Database built successfully!")
