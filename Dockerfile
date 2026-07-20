# Use a lightweight Python 3.10 slim image
FROM python:3.12-slim

# 1. Install system dependencies required for OpenCV and GLib
# We clean up apt lists immediately to keep the image small
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# 2. Set the working directory
WORKDIR /app

# 3. Copy requirements first to leverage Docker cache
COPY requirements-prod.txt .

# 4. Install Python dependencies
# --no-cache-dir reduces image size by not saving downloaded wheel files
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-prod.txt

# 5. Copy the rest of the application code
COPY . .

# 6. Expose the port (Cloud Run ignores this, but it's good practice)
EXPOSE 8080

# 7. Define the command to run the app using Gunicorn
# Cloud Run injects the $PORT environment variable automatically
CMD exec gunicorn --bind :$PORT --workers 1 --threads 1 --timeout 0 app:app
