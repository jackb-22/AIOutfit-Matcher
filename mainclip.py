import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from ultralytics import YOLO
from PIL import Image
import chromadb
from rembg import remove

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()

detector = YOLO("fashion_yolov8.pt")

client = chromadb.PersistentClient(path="./db-clip")
collection = client.get_collection(name="clothes")

def get_embedding(image):
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)

        if not isinstance(features, torch.Tensor):
            if hasattr(features, "image_embeds"):
                features = features.image_embeds
            else:
                features = getattr(features, "pooler_output", features)

    return F.normalize(features, p=2, dim=1).squeeze().tolist()

def detect_and_crop(image_path):
    image = Image.open(image_path).convert("RGB")
    results = detector(image)

    crops_data = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = image.crop((x1, y1, x2, y2))

            class_id = int(box.cls[0])
            label = detector.names[class_id]

            crops_data.append({"crop": crop, "label": label})

    return crops_data

def query(embedding, category_label=None, k=5):
    
    return collection.query(
        query_embeddings=[embedding],
        n_results=k
    )
def run_pipeline(image_path):
    crops_data = detect_and_crop(image_path)
    results = []

    for data in crops_data:
        crop_img = data["crop"]
        label = data["label"]

        # --- THE MAGIC STEP: Remove the background before encoding! ---
        # We convert it back to RGB because rembg adds an alpha (transparency) channel
        clean_crop = remove(crop_img).convert("RGB")

        # Encode the strictly isolated garment
        emb = get_embedding(clean_crop)
        res = query(emb, category_label=label)

        results.append({
            "crop": clean_crop, # Show the transparent crop in the UI!
            "label": label,
            "db_results": res
        })
    return results
