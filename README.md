# Virtual Try-On API

AI-powered virtual try-on service using FastAPI and IDM-VTON model from Hugging Face.

## Features

- 🎨 High-quality virtual try-on using state-of-the-art IDM-VTON model
- 🚀 Fast API built with FastAPI
- 🖼️ Support for multiple garment categories (upper body, lower body, dresses)
- 🔧 Configurable inference parameters
- 📊 Interactive API documentation
- 🐳 GPU and CPU support

## Requirements

- Python 3.9+
- CUDA-compatible GPU (recommended) or CPU
- 8GB+ RAM (16GB+ recommended)
- ~10GB disk space for model cache

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/iqrannwl/virtual-tryon.git
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment

```bash
cp .env.example .env
```

Edit `.env` to configure:
- `DEVICE`: Set to `cuda` for GPU or `cpu` for CPU
- `MODEL_NAME`: Model to use (default: `yisol/IDM-VTON`)
- Other settings as needed

## Usage

### Start the API server

```bash
python main.py
```

Or using uvicorn directly:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### Interactive API Documentation

Visit `http://localhost:8000/docs` for interactive Swagger UI documentation.

### API Endpoints

#### Health Check
```bash
curl http://localhost:8000/api/v1/health
```

#### Virtual Try-On
```bash
curl -X POST http://localhost:8000/api/v1/tryon \
  -F "person_image=@person.jpg" \
  -F "garment_image=@garment.jpg" \
  -F "category=upper_body" \
  -F "num_inference_steps=30" \
  -F "guidance_scale=2.0" \
  -o result.jpg
```

#### List Models
```bash
curl http://localhost:8000/api/v1/models
```

## API Parameters

### Try-On Endpoint (`/api/v1/tryon`)

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `person_image` | File | Yes | - | Image of the person (JPG/PNG) |
| `garment_image` | File | Yes | - | Image of the garment (JPG/PNG) |
| `category` | String | No | `upper_body` | Garment category: `upper_body`, `lower_body`, or `dresses` |
| `num_inference_steps` | Integer | No | 30 | Number of denoising steps (10-50) |
| `guidance_scale` | Float | No | 2.0 | Guidance scale for generation (1.0-5.0) |

## Response Format

### Success Response

```json
{
  "success": true,
  "message": "Virtual try-on completed successfully",
  "result_image": "base64_encoded_image_data...",
  "processing_time": 5.23
}
```

### Error Response

```json
{
  "success": false,
  "error": "ValidationError",
  "message": "Invalid image format",
  "details": "Only JPG and PNG formats are supported"
}
```

## Configuration

Edit `.env` file to customize:

```env
# Model Configuration
MODEL_NAME=yisol/IDM-VTON
DEVICE=cuda  # Use 'cuda' for GPU or 'cpu' for CPU
MODEL_CACHE_DIR=./model_cache

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
MAX_UPLOAD_SIZE=10485760  # 10MB in bytes
ALLOWED_EXTENSIONS=jpg,jpeg,png

# Processing Configuration
DEFAULT_IMAGE_SIZE=768
ENABLE_SAFETY_CHECKER=false
```

## Performance Tips

### GPU Acceleration
- Use CUDA-compatible GPU for 10-20x faster inference
- Ensure PyTorch is installed with CUDA support
- Check GPU availability: `python -c "import torch; print(torch.cuda.is_available())"`

### CPU Mode
- CPU inference is significantly slower (minutes per image)
- Reduce `num_inference_steps` to 10-15 for faster results
- Consider using smaller batch sizes

### Memory Optimization
- The model uses ~6-8GB VRAM on GPU
- CPU mode requires ~8-12GB RAM
- Model is automatically cached after first download

## Troubleshooting

### CUDA Out of Memory
- Reduce image size in config
- Enable CPU offloading (already enabled by default)
- Close other GPU-intensive applications

### Slow Performance
- Ensure you're using GPU mode (`DEVICE=cuda`)
- Reduce `num_inference_steps` to 20-25
- Check GPU utilization: `nvidia-smi`

### Model Download Issues
- Ensure stable internet connection
- Model downloads ~5-10GB on first run
- Check `model_cache/` directory for cached files

## Project Structure

```
virtual-tryon/
├── main.py                 # FastAPI application
├── config.py              # Configuration management
├── requirements.txt       # Python dependencies
├── .env.example          # Environment template
├── .gitignore            # Git ignore rules
├── models/
│   ├── __init__.py
│   └── tryon_model.py    # IDM-VTON model wrapper
├── schemas/
│   ├── __init__.py
│   ├── requests.py       # Request schemas
│   └── responses.py      # Response schemas
└── utils/
    ├── __init__.py
    └── image_processing.py  # Image utilities
```

## Model Information

This API uses **IDM-VTON** (Improving Diffusion Models for Authentic Virtual Try-On):
- Model: `yisol/IDM-VTON`
- Source: Hugging Face
- Paper: [IDM-VTON](https://arxiv.org/abs/2403.05139)
- License: Check model card on Hugging Face

## License

This project is provided as-is for educational and research purposes. Please check the IDM-VTON model license on Hugging Face for commercial use restrictions.

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review API documentation at `/docs`
3. Check model documentation on Hugging Face

## Acknowledgments

- IDM-VTON model by Yisol
- FastAPI framework
- Hugging Face Diffusers library
