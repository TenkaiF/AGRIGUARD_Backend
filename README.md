# AGRIGUARD Backend API

AGRIGUARD is an Android-based expert system designed for real-time plant disease identification and actionable recovery guidance for rural farmers in Malaysia.

This repository contains the lightweight FastAPI backend server hosting the trained YOLOv8 Nano Classification model (`best.pt`).

---

## Architecture & Tech Stack

- Framework FastAPI (Python 3.10+)
- Server Uvicorn
- AI Engine PyTorch  Ultralytics YOLOv8 Nano (`yolov8n-cls`)
- Target Diseases Potato Early Blight, Potato Late Blight, Healthy Potato Leaves
- Localization Structured JSON supporting English (EN) and Bahasa Melayu (BM)

---

## API Endpoint

### `POST predict`
Accepts a leaf image payload from the mobile app and returns diagnostic details, layman certainty ratings, severity levels, and actionable daily recovery steps.

#### Form Data Parameters
- `file` Image file (`UploadFile`)
- `crop_type` Target crop string (`potato`, `tomato`, `sawi`)

#### Sample JSON Response
```json
{
  raw_class Potato___Early_blight,
  selected_crop potato,
  confidence_score 98.45,
  disease_name {
    en Potato Early Blight,
    bm Hawar Awal Kentang
  },
  certainty_level {
    en Very Certain,
    bm Sangat Yakin
  },
  severity {
    en Moderate,
    bm Sederhana
  },
  summary {
    en Fungal infection causing dark concentric spots on older leaves.,
    bm Jangkitan kulat yang menyebabkan bintik-bintik gelap bulat pada daun tua.
  },
  treatment_today {
    en [
      Remove and destroy infected lower leaves immediately.,
      Avoid overhead irrigation to keep leaves dry.,
      Apply copper-based fungicide spray if rainfall is not expected.
    ],
    bm [
      Buang dan musnahkan daun bawah yang dijangkiti dengan segera.,
      Elakkan penyiraman dari atas untuk memastikan daun sentiasa kering.,
      Sembur racun kulat berasaskan kuprum jika hujan tidak dijangka.
    ]
  }
}