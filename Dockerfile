# 1. Base Image
FROM python:3.9-slim

# 2. Create a user with ID 1000 (Required by Hugging Face)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# 3. Set working directory
WORKDIR /app

# 4. Copy requirements and install
COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# 5. Copy the rest of the application
COPY --chown=user . /app

# 6. Hugging Face expects port 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]