# Wildfire Detection Pipeline

End-to-end PyTorch pipeline for binary wildfire classification from satellite imagery,
using a frozen ResNet50 backbone + a trained classification head, with Grad-CAM explainability.

## Setup

1. Download the Kaggle "Wildfire Prediction Dataset" and extract it so it matches:
   ```
   dataset/
       train/nowildfire/ ...
       train/wildfire/   ...
       valid/nowildfire/ ...
       valid/wildfire/   ...
       test/nowildfire/  ...
       test/wildfire/    ...
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Run

```
python main.py --mode all --data_dir dataset --epochs 10
```

Modes: `train`, `evaluate`, `explain`, `all` (default).
See `python main.py --help` for all flags.
