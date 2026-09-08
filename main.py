from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
import PIL.Image as Image
import io

app = FastAPI()

# تحميل النموذج
model = YOLO("model_1_best.pt")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
    
    # إجراء التنبؤ
    results = model(image)
    
    # استخراج النتائج
    detections = []
    for r in results:
        for box in r.boxes:
            detections.append({
                "class": int(box.cls[0]),
                "confidence": float(box.conf[0]),
                "box": box.xyxy[0].tolist()
            })
            
    return {"status": "success", "detections": detections}