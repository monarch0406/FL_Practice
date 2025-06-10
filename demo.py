import gradio as gr
import torch
from torchvision import models, transforms
from PIL import Image
import os
from utils.data_model import GrayscaleToRgb


# List available weight files
def list_model_weights():
    files = os.listdir(WEIGHTS_DIR)
    weight_files = [f for f in files if f.endswith(".pth")]
    return weight_files

# Load model with given weights
def load_model(weight_file):
    model = models.resnet18()  # Assume architecture fixed for now (you can improve later)
    weight_path = os.path.join(WEIGHTS_DIR, weight_file)
    model = torch.load(weight_path, map_location="cpu")
    model.eval()
    return model

# Inference function
def infer(image, weight_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(weight_file)
    model.to(device)

    # Preprocess image
    img_tensor = preprocess(image).unsqueeze(0)  # add batch dimension
    img_tensor = img_tensor.to(device)

    # Forward pass
    with torch.no_grad():
        output, _ = model(img_tensor)
        pred_idx = output.argmax(dim=1).item()

    return f"Prediction: {labels[pred_idx % len(labels)]}"



# Directory where your weights are stored
WEIGHTS_DIR = "./weights"

# Preprocessing
preprocess = transforms.Compose([
    GrayscaleToRgb(),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# {0: "anger", 1: "happy", 2: "neutral"}
# labels = ["anger", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
labels = ["anger", "happy", "neutral"]


# Gradio App
with gr.Blocks() as demo:
    gr.Markdown("# 🖼️ Semtiment Analysis Demo")

    with gr.Row():
        image_input = gr.Image(type="pil", label="Input Image", image_mode="L")
        weight_choice = gr.Dropdown(choices=list_model_weights(), label="Select Model Weights")

    predict_button = gr.Button("Run Inference")
    prediction_output = gr.Textbox(label="Prediction Result")

    predict_button.click(
        infer,
        inputs=[image_input, weight_choice],
        outputs=prediction_output
    )

demo.launch()
