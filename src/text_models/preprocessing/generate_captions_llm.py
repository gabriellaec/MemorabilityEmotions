import ollama
import json
import time
import os
import pandas as pd
import re

IMAGE_DIR = "/home/gabriella/Documents/MemorabilityEmotions/lamem_images/original"
START_IDX = 0
LIMIT = None

OUTPUT_DIR = "llm_captions_results"
BATCH_SIZE = 100

EMOTIONS = [
    "happiness",
    "sadness",
    "fear",
    "anger",
    "disgust",
    "surprise",
]

MODELS = [
    "llava",
]

PROMPT = f"""
You are an advanced image analysis model.

Classify the primary emotion expressed in this image using only one of the following:
{EMOTIONS}

If the emotion is unclear, choose the closest one. Do not use any label outside this list.

Then generate a concise but rich description that adds context, including setting, subjects, and possible situation.

Respond strictly in valid JSON with this structure:
{{
  "emotion": "...",
  "confidence": 0.0,
  "description": "..."
}}
"""

def clean_json(text):
    if text is None:
        return None
    text = text.strip()
    text = re.sub(r"```json", "", text)
    text = re.sub(r"```", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return None
    candidate = text[start:end+1]
    candidate = re.sub(r",\s*([\]}])", r"\1", candidate)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None

def extract_emotion_fallback(text, valid_emotions):
    if not text:
        return None
    lower = text.lower()
    for emotion in valid_emotions:
        if re.search(rf"\b{emotion}\b", lower):
            return emotion
    return None

def run_model(model, image_path):
    start = time.time()
    num_predict = 2048 if model in ("moondream") else 1024
    try:
        response = ollama.chat(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": PROMPT,
                    "images": [image_path],
                }
            ],
            options={"num_predict": num_predict},
        )
        content = response["message"]["content"]
        latency = time.time() - start
        parsed = clean_json(content)
        DESCRIPTION_ALIASES = ("description", "explain", "explanation", "context", "caption")
        if parsed is not None:
            emotion = parsed.get("emotion", "").lower().strip()
            if emotion not in EMOTIONS:
                emotion = None
            description = next(
                (parsed[k] for k in DESCRIPTION_ALIASES if k in parsed), None
            )
            return {
                "emotion": emotion or None,
                "confidence": parsed.get("confidence"),
                "description": description,
                "raw": None,
                "latency": latency,
            }
        else:
            return {
                "emotion": extract_emotion_fallback(content, EMOTIONS),
                "confidence": None,
                "description": None,
                "raw": content,
                "latency": latency,
            }
    except Exception as e:
        return {
            "emotion": None,
            "confidence": None,
            "description": None,
            "raw": str(e),
            "latency": None,
        }

os.makedirs(OUTPUT_DIR, exist_ok=True)

image_files = sorted([
    os.path.join(IMAGE_DIR, f)
    for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
])

def get_processed_images():
    processed = set()
    batch_files = sorted([
        f for f in os.listdir(OUTPUT_DIR)
        if f.startswith("batch_") and f.endswith(".parquet")
    ])
    for bf in batch_files:
        df = pd.read_parquet(os.path.join(OUTPUT_DIR, bf))
        processed.update(df["image"].unique())
    return processed, len(batch_files)

processed_images, existing_batches = get_processed_images()

remaining_images = [
    img for img in image_files
    if os.path.basename(img) not in processed_images
]

remaining_images = remaining_images[START_IDX:]

if LIMIT is not None:
    remaining_images = remaining_images[:LIMIT]

len_remaining = len(remaining_images)
rows = []
batch_idx = existing_batches
images_in_batch = 0

for i, image_path in enumerate(remaining_images):
    image_name = os.path.basename(image_path)
    for model in MODELS:
        result = run_model(model, image_path)
        print(f"{i}/{len_remaining} | Model: {model}, Image: {image_name}, Emotion: {result['emotion']}, Confidence: {result['confidence']}")
        # print(result)
        print()
        rows.append({
            "image": image_name,
            "model": model,
            "emotion": result["emotion"],
            "confidence": result["confidence"],
            "description": result["description"],
            "latency": result["latency"],
            "timestamp": time.time()
        })
    images_in_batch += 1
    if images_in_batch >= BATCH_SIZE:
        df_batch = pd.DataFrame(rows)
        batch_path = os.path.join(
            OUTPUT_DIR,
            f"batch_{str(batch_idx).zfill(4)}.parquet"
        )
        df_batch.to_parquet(batch_path, index=False)
        print(f"Saved batch {batch_idx} with {len(rows)} records to {batch_path}")
        rows = []
        images_in_batch = 0
        batch_idx += 1

if rows:
    df_batch = pd.DataFrame(rows)
    batch_path = os.path.join(
        OUTPUT_DIR,
        f"batch_{str(batch_idx).zfill(4)}.parquet"
    )
    df_batch.to_parquet(batch_path, index=False)
