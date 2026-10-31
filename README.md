# Crop Disease Classifier

Image classifier for plant leaf diseases, trained on 12 classes of the
[PlantVillage dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
and served as a FastAPI app on Vercel.

**Live demo: https://crop-disease-classifier.vercel.app**

## Model comparison (held-out test set, 1,952 images)

| Model | Accuracy | Macro F1 | Params (M) | Size (MB) | CPU latency (ms) |
|---|---|---|---|---|---|
| small_cnn | 0.936 | 0.927 | 0.39 | 1.6 | 83.6 |
| **mobilenet_v3_small** | **0.998** | **0.998** | 1.53 | 6.2 | 35.3 |
| resnet18 | 0.992 | 0.987 | 11.18 | 44.8 | 55.9 |
| regnet_x_400mf | 0.998 | 0.998 | 5.10 | 20.7 | 25.0 |

**MobileNetV3-Small** is deployed: it has the best macro F1 and a small footprint. It is exported
to ONNX (`model/mobilenet_v3_small.onnx`) so the server only needs `onnxruntime`, not PyTorch.

### Classes
Apple — Black rot, Cedar apple rust · Cherry — healthy · Corn — Cercospora/Gray leaf spot, Common rust ·
Grape — healthy · Orange — Huanglongbing (citrus greening) · Peach — healthy · Potato — Late blight ·
Raspberry — healthy · Tomato — Bacterial spot, Yellow leaf curl virus

## Project layout

```
app.py                 FastAPI app (web UI at /, API at /api/*)
public/index.html      Upload page
model/                 Deployed ONNX model
train.py               Data prep + training of all 4 models
eda.ipynb, eval.ipynb  Exploration and evaluation
export_onnx.py         Converts mobilenet_v3_small.pt -> ONNX
data/pull_data.py      Downloads the dataset via kagglehub
```

## API

```bash
curl -F "file=@leaf.jpg" "https://crop-disease-classifier.vercel.app/api/predict?top_k=3"
```

```json
{
  "prediction": {"label": "Tomato___Bacterial_spot", "crop": "Tomato", "condition": "Bacterial spot", "confidence": 0.998},
  "top_k": [ ... ]
}
```

`GET /api/health` returns the model name and class list.

## Run locally

```bash
pip install -r requirements.txt uvicorn
uvicorn app:app --reload
# open http://localhost:8000
```

## Training

```bash
pip install -r requirements-train.txt
python train.py           # trains and saves <model>.pt
python export_onnx.py     # exports the best model to model/
```

## Deploy

Vercel detects the FastAPI `app` in `app.py` and installs `requirements.txt`.
Import the GitHub repo at https://vercel.com/new (no settings needed), or run `vercel --prod`.

## Limitations
PlantVillage images are lab photos of single leaves on plain backgrounds; accuracy on field photos
will be lower. The model only knows these 12 classes and will always pick one of them.
