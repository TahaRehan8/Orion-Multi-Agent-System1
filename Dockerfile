FROM python:3.10-slim

# Hugging Face Spaces require the app to run as a non-root user with UID 1000
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Switch to root to change ownership, then back to user
USER root
RUN chown -R user:user /app
USER user

# Copy requirements and install dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY --chown=user . .

# Expose port 7860 as required by Hugging Face
EXPOSE 7860

# Run the FastAPI application
CMD ["uvicorn", "backend.api:app", "--host", "0.0.0.0", "--port", "7860"]
