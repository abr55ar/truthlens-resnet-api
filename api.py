import os
import io

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from torchvision import models, transforms


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "deepfake_resnet50_balanced.pth"
)
IMAGE_SIZE = 224

print("Loading TruthLens ResNet50...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model.to(device)
model.eval()

print(f"Model loaded: {MODEL_PATH}")
print(f"Device: {device}")


# ============================================================
# OPENCV FACE DETECTOR
# ============================================================

CASCADE_PATH = os.path.join(os.path.dirname(__file__), "haarcascade_frontalface_default.xml")

face_detector = cv2.CascadeClassifier(CASCADE_PATH)

if face_detector.empty():
    raise RuntimeError("Could not load OpenCV Haar face detector.")

print("OpenCV face detector loaded.")


# ============================================================
# IMAGE TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="TruthLens ResNet50 API",
    version="1.2.0",
    description="TruthLens AI deepfake detection API using ResNet50."
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# FACE DETECTION
# ============================================================

def detect_faces(image_rgb):

    gray = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2GRAY
    )

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    return faces


def crop_largest_face(image_rgb, faces):

    if len(faces) == 0:
        return None

    # Select largest detected face
    largest = max(
        faces,
        key=lambda box: box[2] * box[3]
    )

    x, y, w, h = largest

    # Add padding around face
    padding = 0.20

    px = int(w * padding)
    py = int(h * padding)

    x1 = max(0, x - px)
    y1 = max(0, y - py)

    x2 = min(image_rgb.shape[1], x + w + px)
    y2 = min(image_rgb.shape[0], y + h + py)

    face_crop = image_rgb[y1:y2, x1:x2]

    return face_crop


# ============================================================
# MODEL PREDICTION
# ============================================================

def predict_image(image):

    image_rgb = np.array(image.convert("RGB"))

    faces = detect_faces(image_rgb)

    face_count = len(faces)

    if face_count == 0:

        return {
            "prediction": "NO_FACE",
            "confidence": 0.0,
            "score": 0.0,
            "face_detected": False,
            "face_count": 0
        }

    face_crop = crop_largest_face(
        image_rgb,
        faces
    )

    if face_crop is None:
        raise RuntimeError("Face detected but crop failed.")

    face_image = Image.fromarray(face_crop)

    tensor = transform(face_image)

    tensor = tensor.unsqueeze(0).to(device)

    with torch.no_grad():

        output = model(tensor)

        probabilities = torch.softmax(
            output,
            dim=1
        )[0]

    real_probability = probabilities[0].item()
    fake_probability = probabilities[1].item()

    if fake_probability >= real_probability:

        prediction = "FAKE"
        confidence = fake_probability

    else:

        prediction = "REAL"
        confidence = real_probability

    return {
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "score": round(confidence * 100, 2),
        "face_detected": True,
        "face_count": face_count
    }


# ============================================================
# ROUTES
# ============================================================

@app.get("/")
def root():

    return {
        "service": "TruthLens ResNet50 API",
        "status": "online",
        "model": "ResNet50",
        "version": "1.2.0",
        "device": str(device)
    }


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": True,
        "device": str(device)
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    if not file.content_type:
        raise HTTPException(
            status_code=400,
            detail="File type could not be determined."
        )

    allowed_types = [
        "image/jpeg",
        "image/png",
        "image/webp"
    ]

    if file.content_type not in allowed_types:

        raise HTTPException(
            status_code=400,
            detail="Only JPG, PNG and WEBP images are supported."
        )

    try:

        contents = await file.read()

        image = Image.open(
            io.BytesIO(contents)
        ).convert("RGB")

        result = predict_image(image)

        return {
            **result,
            "model": "ResNet50",
            "model_version": "balanced-v1",
            "device": str(device)
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ============================================================
# STARTUP
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000
    )
