# Custom Sentiment Analysis System

This system allows you to train a custom sentiment analysis model using DistilBERT on your own labeled dataset. It supports multi-label probability outputs to capture complex "over the bridge" emotions.

## Project Structure

- `train.py`: Fine-tunes the model on a labeled `.txt` dataset.
- `predict.py`: CLI tool for sentiment prediction.
- `api.py`: FastAPI server providing a `/predict` endpoint.
- `model_service.py`: Core logic for loading the model and running inference.
- `preprocess.py`: Text preprocessing module.
- `ui/app.py`: Streamlit-based web interface.
- `data/`: Directory for training data.
- `models/`: Directory where the trained model is saved.
- `outputs/`: Directory for training logs and metrics.

## Installation

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Training

To train the model on the default dataset (`data/train.txt`):
```bash
python train.py
```

To train on a custom file:
```bash
python train.py --data path/to/your_data.txt --epochs 5
```

The dataset should be a `.txt` file with lines in the format: `text;label`.

## Usage

### CLI Prediction
```bash
python predict.py "I feel so amazing today!"
```

### FastAPI Server
Start the server:
```bash
python api.py
```
The API will be available at `http://localhost:8000`. You can send a POST request to `/predict`:
```bash
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"text": "I am feeling a bit uncertain."}'
```

### Streamlit UI
Start the UI:
```bash
streamlit run ui/app.py
```

## Dataset Format
The system expects a semicolon-separated `.txt` file:
```
I love this;joy
I am so angry;anger
It is okay;neutral
```
The system automatically detects all unique labels and configures the model accordingly.


## Deployment on Free Hosting (Fast + Low Download Size)

If you see very large downloads like `nvidia_cudnn_cu12` during `pip install`, that means pip is pulling GPU CUDA wheels for `torch`.
On free hosting, prefer CPU-only builds.

### 1) Use CPU-only requirements
For backend/API deployments use:

```bash
pip install -r requirements.cpu.txt
```

This pins CPU-only PyTorch and avoids multi-GB CUDA package downloads.

### 2) Host model files on Hugging Face Model Hub
1. Train locally: `python train.py`
2. Upload `models/sentiment_model/` contents to a HF model repo.
3. In backend startup, download model files into `./models/sentiment_model` before starting API.

### 3) Host backend for free (Render)
1. Push this repo to GitHub.
2. Create a Render **Web Service**.
3. Build command:
   ```bash
   pip install -r requirements.cpu.txt
   ```
4. Start command:
   ```bash
   uvicorn api:app --host 0.0.0.0 --port $PORT
   ```
5. Ensure model files exist at `./models/sentiment_model` (download from HF during startup).

### 4) Host frontend for free (Hugging Face Spaces)
1. Create a **Streamlit** Space.
2. Use lightweight UI dependencies:
   ```bash
   pip install -r requirements.ui.txt
   ```
3. Call your Render backend `/predict` endpoint from the UI.
4. Add backend URL in Space Secrets (example: `API_URL`).

### 5) Speed tips
- Keep backend and model in same region.
- Keep backend warm (free tiers may sleep).
- Add confidence threshold (for example 0.70) so uncertain predictions are not shown as final.
