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
