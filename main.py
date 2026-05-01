import torch
import torch.nn.functional as F
from ultralytics import YOLO
from PIL import Image
import chromadb
from transformers import AutoImageProcessor, AutoModel
from rembg import remove

detector = YOLO("fashion_yolov8.pt")

client = chromadb.PersistentClient(path="./db")
collection = client.get_collection(name="clothes")

processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
model = AutoModel.from_pretrained('facebook/dinov2-base')
model.eval()

USE_BACKGROUND_REMOVAL = False

def get_embedding(image):
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        vector = outputs.last_hidden_state[:, 0, :].squeeze().tolist()
    return vector

def detect_and_crop(image_path):
    image = Image.open(image_path).convert("RGB")
    results = detector(image, conf=0.5, iou=0.4)

    crops_data = []

    seen_shoes = set()
    shoe_keywords = {"shoe", "shoes", "sneaker", "sneakers", "boot", "boots", "footwear"}

    for r in results:
        for box in r.boxes:
            class_id = int(box.cls[0])
            label = detector.names[class_id].lower()

            if label in shoe_keywords:
                if label in seen_shoes:
                    continue
                seen_shoes.add(label)

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = image.crop((x1, y1, x2, y2))
            crops_data.append({"crop": crop, "label": label})

    return crops_data

YOLO_TO_CATEGORY = {
    "shirt": "Clothing", "t-shirt": "Clothing", "top": "Clothing",
    "pants": "Clothing", "jeans": "Clothing", "shorts": "Clothing",
    "jacket": "Clothing", "coat": "Clothing", "dress": "Clothing",

    "shoe": "Shoes", "sneaker": "Shoes", "boot": "Shoes",
    "shoes": "Shoes", "sneakers": "Shoes", "boots": "Shoes", "footwear": "Shoes",

    "bag": "Accessories", "hat": "Accessories", "cap": "Accessories",
    "bags": "Accessories", "hats": "Accessories"
}

LABEL_MAP = {
    "Clothing": [
        "T-Shirt", "Dress", "Outerwear", "Skirt", "Jeans", "Sweater",
        "Blazer", "Top", "Henley", "Hoodie/Crew/1/4 Zips", "Tank/Camisole",
        "Pants", "Button down", "Shorts", "Athleisure", "Polo", "Vest",
        "Jersey", "Rugby", "Intimates", "Other"
    ],
    "Shoes": [
        "Shoes", "Boots", "Sneakers", "Shoe", "Boot", "Sneaker",
        "Footwear"
    ],
    "Accessories": [
        "Bags", "Accessories", "Hats", "Hat", "Beanie", "Ties", "Scarf"
    ]
}
def query(embedding, category_label=None, k=5):
    if category_label and category_label in LABEL_MAP:
        allowed_categories = LABEL_MAP[category_label]

        res = collection.query(
            query_embeddings=[embedding],
            n_results=k,
            where={"category": {"$in": allowed_categories}}
        )

        if len(res.get("ids", [[]])[0]) == 0:
            res = collection.query(
                query_embeddings=[embedding],
                n_results=k,
                where={"Category": {"$in": allowed_categories}}
            )

        return res

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

        if USE_BACKGROUND_REMOVAL:
            try:
                processed_crop = remove(crop_img).convert("RGB")
            except:
                processed_crop = crop_img.convert("RGB")
        else:
            processed_crop = crop_img.convert("RGB")

        emb = get_embedding(processed_crop)

        mapped_category = YOLO_TO_CATEGORY.get(label.lower(), None)

        print(f"DEBUG -> YOLO Found: '{label}', Mapped to: '{mapped_category}'")

        res = query(emb, category_label=mapped_category)

        results.append({
            "crop": processed_crop,
            "label": label,
            "mapped_category": mapped_category,
            "db_results": res
        })

    return results
