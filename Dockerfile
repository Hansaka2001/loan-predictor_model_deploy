# 1. Base Image: Use a lightweight Python version
FROM python:3.9-slim

# 2. Set working directory inside the container
WORKDIR /app

# 3. Copy requirements first (for caching optimization)
COPY requirements.txt .

# 4. Install dependencies
# We use --no-cache-dir to keep the image small
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of your application code (main.py, loan_model.pkl, etc.)
COPY . .

# 6. Expose the port the app runs on
EXPOSE 8000

# 7. Command to run the application
# host 0.0.0.0 is crucial for Docker containers to accept external connections
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]