# Food Image Classification

A Flask web app for uploading a food photo, running it through one of three
models (Custom CNN / VGG16 / ResNet50), and comparing nutrition facts +
model metrics for the **actual** class you select vs. the **predicted** class.

## Project structure

```
food-classifier/
├── app.py                    # Flask backend + inference
├── requirements.txt
├── templates/
│   └── index.html
├── static/
│   ├── css/style.css
│   └── js/script.js
├── data/
│   ├── classes.json          # class names, in model output order
│   ├── nutrition.json        # nutrition facts per class
│   ├── metrics_cnn.json      # accuracy/precision/recall/f1 per class
│   ├── metrics_vgg16.json
│   └── metrics_resnet50.json
└── models/                   # put your trained weight files here
    ├── cnn_model.h5
    ├── vgg16_model.h5
    └── resnet50_model.h5
```

## Quick start

```bash
pip install -r requirements.txt
python app.py
```

Then open **http://localhost:5000**.

The app runs immediately, even with no model weights present — it falls
back to **demo mode** (a deterministic placeholder prediction) so you can
try the whole flow: upload a picture → pick a model → pick the actual class
from the searchable list on the right → Predict. This lets you review and
tweak the UI before your real models are wired up.

## Connecting your real models

1. Install TensorFlow: `pip install tensorflow` (uncomment it in
   `requirements.txt`).
2. Drop your trained `.h5` files into `models/`:
   - `cnn_model.h5`
   - `vgg16_model.h5`
   - `resnet50_model.h5`
3. Make sure `data/classes.json` lists your classes in **exactly the same
   order** your model's output layer was trained on — the app maps the
   softmax index straight back to this list.
4. Restart the app. As soon as a weight file is found, that model switches
   from demo mode to real inference automatically (see `demo_mode` in the
   `/api/predict` response, also surfaced to the user in the UI).

If your models were trained in PyTorch instead of Keras/TensorFlow, replace
`load_keras_model()` and `run_inference()` in `app.py` with your own
loading/inference code — everything else (routes, response JSON shape,
frontend) stays the same.

## Replacing the sample data

`data/classes.json`, `data/nutrition.json`, and the three
`data/metrics_*.json` files currently ship with **placeholder/sample
values** for the 101 Food-101 classes, so the app is fully demoable out of
the box. Replace them with your real data — same shape:

```jsonc
// nutrition.json
{
  "pizza": {
    "display_name": "Pizza",
    "serving_size": "1 serving (~150g)",
    "calories": 429,
    "protein_g": 10.0,
    "carbs_g": 33.3,
    "fat_g": 3.1,
    "fiber_g": 2.7,
    "sugar_g": 23.8,
    "sodium_mg": 878
  }
}
```

```jsonc
// metrics_cnn.json / metrics_vgg16.json / metrics_resnet50.json
{
  "pizza": { "accuracy": 0.81, "precision": 0.79, "recall": 0.77, "f1_score": 0.78 }
}
```

## Notes

- Images are processed in memory and are **not** saved to disk.
- Max upload size is 10 MB (`MAX_CONTENT_LENGTH` in `app.py`).
- The "Nutrition Facts" cards are styled after the familiar FDA label as a
  deliberate design choice — swap colors/fonts in `static/css/style.css`
  (`:root` variables at the top) to match your own brand.
