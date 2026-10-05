import os
import io
import cv2
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import models, transforms

MODEL_PATH = os.path.join(os.getcwd(), "deepfake_resnet50_balanced.pth")
CASCADE_PATH = os.path.join(os.getcwd(), "haarcascade_frontalface_default.xml")
API_KEY = os.getenv("TRUTHLENS_API_KEY")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(title="TruthLens ResNet50 API", version="1.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE)
)

model.to(DEVICE)
model.eval()

face_detector = cv2.CascadeClassifier(CASCADE_PATH)

if face_detector.empty():
    raise RuntimeError("Failed to load Haar cascade classifier.")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

@app.get("/")
def root():
    return {
        "service": "TruthLens ResNet50 API",
        "status": "online",
        "model": "ResNet50",
        "version": "1.2.0",
        "device": str(DEVICE)
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
        "device": str(DEVICE)
    }

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    x_api_key: str = Header(None)
):
    if not API_KEY:
        raise HTTPException(
            status_code=500,
            detail="API key is not configured"
        )

    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="Empty file"
        )

    try:
        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file"
        )

    image_cv = cv2.cvtColor(
        np.array(image),
        cv2.COLOR_RGB2BGR
    )

    gray = cv2.cvtColor(
        image_cv,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40)
    )

    face_count = len(faces)

    if face_count > 0:
        largest_face = max(
            faces,
            key=lambda rect: rect[2] * rect[3]
        )

        x, y, w, h = largest_face

        padding_x = int(w * 0.20)
        padding_y = int(h * 0.20)

        x1 = max(0, x - padding_x)
        y1 = max(0, y - padding_y)

        x2 = min(
            image_cv.shape[1],
            x + w + padding_x
        )

        y2 = min(
            image_cv.shape[0],
            y + h + padding_y
        )

        face_crop = image_cv[y1:y2, x1:x2]

        face_crop = cv2.cvtColor(
            face_crop,
            cv2.COLOR_BGR2RGB
        )

        face_image = Image.fromarray(face_crop)

        face_detected = True

    else:
        face_image = image
        face_detected = False

    tensor = transform(
        face_image
    ).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(tensor)

        probabilities = torch.softmax(
            output,
            dim=1
        )

        real_probability = probabilities[0, 0].item()
        fake_probability = probabilities[0, 1].item()

    if fake_probability >= 0.5:
        prediction = "FAKE"
        confidence = fake_probability
    else:
        prediction = "REAL"
        confidence = real_probability

    score = confidence * 100

    return {
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "score": round(score, 2),
        "face_detected": face_detected,
        "face_count": face_count,
        "model": "ResNet50",
        "model_version": "balanced-v1",
        "device": str(DEVICE)
    }
