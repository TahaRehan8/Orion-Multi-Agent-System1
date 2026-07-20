import gradio as gr
from backend.api import app as fastapi_app

# Create a minimal Gradio UI to satisfy the Hugging Face Gradio SDK requirements
demo = gr.Interface(
    fn=lambda: "Orion Backend is fully operational!",
    inputs=None,
    outputs="text",
    title="Orion API Backend",
    description="This space hosts the FastAPI backend for the Orion Multi-Agent RAG System. The REST API is available at the root URL."
)

# Mount the Gradio UI at a subpath (/ui) so it doesn't conflict with our FastAPI root "/" endpoints
# The Hugging Face SDK will detect this combined ASGI application and serve it perfectly.
app = gr.mount_gradio_app(fastapi_app, demo, path="/ui")
