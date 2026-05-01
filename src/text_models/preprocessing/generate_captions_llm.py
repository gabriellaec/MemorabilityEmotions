import ollama
import json
import time
import os
import pandas as pd
import re
 
IMAGE_DIR = "/Users/celsocukier/Documents/CompNeuroscience/MemorabilityEmotions/lamem_images/original"
START_IDX = 0
LIMIT = 3
OUTPUT_PATH = "results.parquet"


EMOTIONS = [
    "happiness",
    "sadness",
    "fear",
    "anger",
    "disgust",
    "surprise",
]

MODELS = [
    "llava:13b",
    "llava",
    "moondream",
    # "qwen2.5vl:7b",
    # "minicpm-v",
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
    # Strip markdown fences
    text = re.sub(r"```json", "", text)
    text = re.sub(r"```", "", text)

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        return None

    candidate = text[start:end+1]

    # Fix common model quirks: trailing commas before closing brace/bracket
    candidate = re.sub(r",\s*([\]}])", r"\1", candidate)

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        # Last resort: try json5-style tolerant parse via ast.literal_eval won't work,
        # so fall through to the keyword extractor below
        return None


def extract_emotion_fallback(text, valid_emotions):
    """When JSON parsing fails entirely, scan raw text for a known emotion keyword."""
    if not text:
        return None
    lower = text.lower()
    for emotion in valid_emotions:
        # Match whole word to avoid 'fearful' matching 'fear' unexpectedly
        if re.search(rf"\b{emotion}\b", lower):
            return emotion
    return None


def run_model(model, image_path):
    start = time.time()

    # Models known to truncate — raise their output limit
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
            options={"num_predict": num_predict},  # ← fix moondream truncation
        )

        content = response["message"]["content"]
        latency = time.time() - start
        parsed = clean_json(content)

        # llava:13b returns {"emotion","confidence","explain"} instead of "description"
        # Normalize alternative key names here
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
    

def check_model_supports_images(model):
    """Probe with a tiny dummy call before the real loop."""
    try:
        # Use a 1x1 white pixel PNG (base64 inline) to avoid needing a real file
        import base64
        tiny_png = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwADhQGAWjR9awAAAABJRU5ErkJggg=="
        )
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(tiny_png)
            tmp = f.name
        ollama.chat(model=model, messages=[{"role": "user", "content": "ok", "images": [tmp]}])
        os.unlink(tmp)
        return True
    except Exception as e:
        print(f"[SKIP] {model} does not support images: {e}")
        return False

MODELS = [m for m in MODELS if check_model_supports_images(m)]


print(f"Processing {len(MODELS)} models")

image_files = sorted([
    os.path.join(IMAGE_DIR, f)
    for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
])

if os.path.exists(OUTPUT_PATH):
    df_existing = pd.read_parquet(OUTPUT_PATH)
    processed = set(df_existing["image"].unique())
else:
    df_existing = pd.DataFrame()
    processed = set()

remaining_images = [
    img for img in image_files
    if os.path.basename(img) not in processed
]

remaining_images = remaining_images[START_IDX:]

if LIMIT is not None:
    remaining_images = remaining_images[:LIMIT]

rows = []

for image_path in remaining_images:
    image_name = os.path.basename(image_path)

    for model in MODELS:
        result = run_model(model, image_path)

        print(f"Model: {model}, Image: {image_name}")
        print(result)
        print(result["description"])
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

df_new = pd.DataFrame(rows)

if not df_existing.empty:
    df_final = pd.concat([df_existing, df_new], ignore_index=True)
else:
    df_final = df_new

df_final.to_parquet(OUTPUT_PATH, index=False)

print(df_final.head())
