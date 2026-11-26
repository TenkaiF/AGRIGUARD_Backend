from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import onnxruntime as ort
import numpy as np
from PIL import Image
import io

app = FastAPI(title="AGRIGUARD API", version="3.2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

session = ort.InferenceSession("best.onnx", providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

CLASSES = {
    0: "Potato___Early_blight",
    1: "Potato___Late_blight",
    2: "Potato___healthy"
}

DISEASE_INFO = {
    "Potato___Early_blight": {
        "disease_name": {"en": "Potato Early Blight", "bm": "Hawar Awal Kentang"},
        "severity": {"en": "Moderate", "bm": "Sederhana"},
        "summary": {
            "en": "Fungal infection causing dark concentric spots on older leaves.",
            "bm": "Jangkitan kulat yang menyebabkan bintik-bintik gelap bulat pada daun tua."
        },
        "treatment_today": {
            "en": [
                "Remove and destroy infected lower leaves immediately.",
                "Avoid overhead irrigation to keep leaves dry.",
                "Apply copper-based fungicide spray if rainfall is not expected."
            ],
            "bm": [
                "Buang dan musnahkan daun bawah yang dijangkiti dengan segera.",
                "Elakkan penyiraman dari atas untuk memastikan daun sentiasa kering.",
                "Sembur racun kulat berasaskan kuprum jika hujan tidak dijangka."
            ]
        }
    },
    "Potato___Late_blight": {
        "disease_name": {"en": "Potato Late Blight", "bm": "Hawar Lewat Kentang"},
        "severity": {"en": "High / Critical", "bm": "Bahaya / Kritikal"},
        "summary": {
            "en": "Severe water-soaked lesions that can destroy crops rapidly in damp conditions.",
            "bm": "Luka basah teruk yang boleh memusnahkan tanaman dengan cepat dalam keadaan lembap."
        },
        "treatment_today": {
            "en": [
                "Isolate affected plants immediately.",
                "Apply systemic fungicide (e.g., Mancozeb) immediately.",
                "Do not harvest tubers during wet conditions to prevent rot spread."
            ],
            "bm": [
                "Asingkan tumbuhan yang terjejas dengan serta-merta.",
                "Sembur racun kulat sistemik (contoh: Mancozeb) dengan segera.",
                "Jangan dituai semasa keadaan basah untuk elak penyebaran reput."
            ]
        }
    },
    "Potato___healthy": {
        "disease_name": {"en": "Healthy Plant", "bm": "Tumbuhan Sihat"},
        "severity": {"en": "Healthy", "bm": "Sihat"},
        "summary": {
            "en": "Leaf appears healthy with no signs of fungal or bacterial infection.",
            "bm": "Daun kelihatan sihat tanpa sebarang tanda jangkitan kulat atau bakteria."
        },
        "treatment_today": {
            "en": [
                "Maintain regular watering at soil level.",
                "Continue standard fertilizer application schedule."
            ],
            "bm": [
                "Kekalkan penyiraman berkala pada aras tanah.",
                "Teruskan jadual pemupukan standard."
            ]
        }
    }
}

def get_certainty_labels(confidence_pct: float):
    if confidence_pct >= 80.0:
        return {"en": "Very Certain", "bm": "Sangat Yakin"}
    elif confidence_pct >= 60.0:
        return {"en": "Likely", "bm": "Kemungkinan Besar"}
    else:
        return {"en": "Uncertain", "bm": "Tidak Pasti"}

@app.get("/")
def root():
    return {"status": "AGRIGUARD Cloud API Active"}

@app.post("/predict")
async def predict_leaf_disease(
    file: UploadFile = File(...),
    crop_type: str = Form("potato")
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image.")
    
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # 1. Resize image to 224x224
    image = image.resize((224, 224), Image.Resampling.BILINEAR)
    
    # 2. Scale pixel values to [0.0, 1.0]
    img_np = np.array(image, dtype=np.float32) / 255.0
    
    # 3. Transpose HWC [224, 224, 3] -> CHW [3, 224, 224] & Add Batch Dim [1, 3, 224, 224]
    img_np = np.transpose(img_np, (2, 0, 1))
    img_np = np.expand_dims(img_np, axis=0)
    
    # 4. Execute ONNX Inference
    outputs = session.run(None, {input_name: img_np})
    raw_output = outputs[0][0]
    
    # Direct readout (ONNX graph already outputs class probabilities)
    top1_index = int(np.argmax(raw_output))
    confidence = float(raw_output[top1_index]) * 100
    
    print(f"DEBUG: Output={raw_output}, TopIndex={top1_index}, TrueConf={confidence:.2f}%")
    
    raw_class_name = CLASSES.get(top1_index, "Potato___healthy")
    
    details = DISEASE_INFO.get(raw_class_name, {
        "disease_name": {"en": f"{crop_type.title()} Condition", "bm": f"Keadaan {crop_type.title()}"},
        "severity": {"en": "Moderate", "bm": "Sederhana"},
        "summary": {"en": "Condition detected.", "bm": "Keadaan dikesan."},
        "treatment_today": {"en": ["Monitor plant health."], "bm": ["Pantau kesihatan tumbuhan."]}
    })
    
    certainty = get_certainty_labels(confidence)
    
    return {
        "raw_class": raw_class_name,
        "selected_crop": crop_type,
        "confidence_score": round(confidence, 2),
        "disease_name": details["disease_name"],
        "certainty_level": certainty,
        "severity": details["severity"],
        "summary": details["summary"],
        "treatment_today": details["treatment_today"]
    }
