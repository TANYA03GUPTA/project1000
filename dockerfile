# Use a CUDA and cuDNN base image with Python as the parent image
FROM nvidia/cuda:11.0-cudnn8-runtime-ubuntu18.04

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install the Python dependencies and GPU-enabled TensorFlow 2.12.0
RUN pip install --no-cache-dir tensorflow-gpu==2.12.0 -r requirements.txt

# Copy the Flask app code
COPY . .

# Expose the port on which the app will run
EXPOSE 5002

# Start the Flask app
CMD ["python", "app.py"]
