# Use an official Python image
FROM python:3.10-slim

# Set work directory
WORKDIR /model_railway

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app and model files
COPY main.py .
COPY rf_model.pkl .
COPY rf_scaler.pkl .

# Run the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
