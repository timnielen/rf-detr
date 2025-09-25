from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import uvicorn
import os
import io
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rfdetr import RFDETRNano
import pandas as pd

app = FastAPI()

# Allow Nuxt frontend
origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,          # or ["*"] for all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


model = RFDETRNano(lite_refpoint_refine=False, pretrain_weights="outputs/output_dual_4_points_weighted_loss_more_data/checkpoint_best_total.pth", num_boxes_per_query=2, dec_n_points=4, device="cpu")
IMAGE_DIR = "../playdarts/my_unlabelled/vids"
subdirs = [d for d in os.listdir(IMAGE_DIR) if os.path.isdir(os.path.join(IMAGE_DIR, d))]

@app.get("/folders")
async def get_folders():
    paths = [os.path.join(IMAGE_DIR, d, "labels.pkl") for d in subdirs]
    labels = [pd.read_pickle(path).to_dict(orient="records") for path in paths if os.path.isfile(path)]
    return [{"name": d, "annotations": label} for d, label in zip(subdirs, labels)]

@app.get("/images/{folder}/{image_name}")
async def get_image(folder: str, image_name: str):
    if folder not in subdirs:
        raise HTTPException(status_code=404, detail="Folder not found")
    images = [f for f in os.listdir(os.path.join(IMAGE_DIR, folder)) if os.path.isfile(os.path.join(IMAGE_DIR, folder, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp'))]
    if image_name not in images:
        raise HTTPException(status_code=404, detail="Image not found")
    
    return FileResponse(os.path.join(IMAGE_DIR, folder, image_name))

@app.get("/predict/{folder}/{image_name}")
async def predict(folder: str, image_name: str):
    if folder not in subdirs:
        raise HTTPException(status_code=404, detail="Folder not found")
    images = [f for f in os.listdir(os.path.join(IMAGE_DIR, folder)) if os.path.isfile(os.path.join(IMAGE_DIR, folder, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp'))]
    if image_name not in images:
        raise HTTPException(status_code=404, detail="Image not found")

    img = Image.open(os.path.join(IMAGE_DIR, folder, image_name))
    results = model.predict(img, threshold=0.3)
    
    xyxy = results.xyxy.reshape(-1, 8).tolist()
    labels = results.class_id[::2].tolist()
    confidences = results.confidence[::2].tolist()
    return JSONResponse(content={"xyxy": xyxy, "labels": labels, "confidence": confidences})