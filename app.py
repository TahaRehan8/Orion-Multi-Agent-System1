import sys

# --- HUGGING FACE SPACES HACK ---
# Hugging Face's ZeroGPU SDK forces an old version of Gradio (4.36.1) that looks for 'HfFolder'.
# However, Hugging Face recently removed 'HfFolder' from their 'huggingface_hub' library!
# We monkey-patch it here before importing Gradio so the server doesn't crash on startup.
try:
    import huggingface_hub
    if not hasattr(huggingface_hub, 'HfFolder'):
        class HfFolder:
            pass
        huggingface_hub.HfFolder = HfFolder
except Exception:
    pass

import gradio as gr
from backend.api import app as fastapi_app

# Create a dummy Gradio interface just to satisfy Hugging Face Spaces
# It will never actually be seen because the Next.js frontend calls the API directly
def dummy_function():
    return "Orion Multi-Agent RAG API is running on Hugging Face Spaces!"

demo = gr.Interface(
    fn=dummy_function, 
    inputs=None, 
    outputs="text",
    title="Orion Backend API",
    description="This is the backend API. Please use the Vercel frontend to interact with this service."
)

# Mount the FastAPI app onto the Gradio interface
app = gr.mount_gradio_app(fastapi_app, demo, path="/ui")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
