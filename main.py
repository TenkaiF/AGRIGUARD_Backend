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
    image = image.resize((224, 224))
    
    # 2. Convert PIL Image to float32 array normalized to [0, 1]
    img_array = np.array(image, dtype=np.float32) / 255.0
    
    # 3. Apply ImageNet Mean & Std Normalization (Standard PyTorch weights format)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_array = (img_array - mean) / std
    
    # 4. Reorder dimensions from HWC to CHW [3, 224, 224] and add batch dimension [1, 3, 224, 224]
    img_array = np.transpose(img_array, (2, 0, 1))
    img_array = np.expand_dims(img_array, axis=0)
    
    # 5. Run ONNX Inference
    outputs = session.run(None, {input_name: img_array})
    probs = outputs[0][0]
    
    # 6. Apply Softmax to raw logits
    exp_probs = np.exp(probs - np.max(probs))
    softmax_probs = exp_probs / exp_probs.sum()
    
    top1_index = int(np.argmax(softmax_probs))
    confidence = float(softmax_probs[top1_index]) * 100
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
