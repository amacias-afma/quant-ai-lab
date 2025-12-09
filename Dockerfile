# Use a lightweight Python base image
FROM python:3.9-slim

# 1. Install system dependencies required for C++ based math libraries (numpy, arch)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the rest of the application code
COPY . .

# 5. Define the command to run the app using Gunicorn (Production Server)
# This assumes your main flask app is in 'src/app.py' and the variable is 'server' or 'app'
# We will create a dummy app next to make this pass.
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 src.app:app