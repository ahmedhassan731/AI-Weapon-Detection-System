FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV in Linux
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (CPU PyTorch + requirements)
COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose Gradio default port
EXPOSE 7860

# Run Gradio Web UI
CMD ["python", "gradio_app.py"]
