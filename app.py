import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import gradio as gr
import numpy as np
import cv2
import matplotlib.cm as cm
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from datetime import datetime
import os
import time
import traceback

# Device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class labels
class_labels = [
    'actinic_keratoses', 'basal_cell_carcinoma', 'benign_keratosis_like_lesions',
    'dermatofibroma', 'melanoma', 'nevus', 'vascular_lesions'
]

# Load model
def load_model():
    print("🔍 Loading model...")
    model = models.efficientnet_b0(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(num_ftrs, 7)
    )
    model.load_state_dict(torch.load("best_skin_model.pth", map_location=device))
    model.to(device)
    model.eval()
    print("✅ Model loaded successfully.")
    return model

model = load_model()

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Prediction function
def predict_class(img_pil):
    image_tensor = transform(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
        pred_idx = np.argmax(probs)
        pred_label = class_labels[pred_idx]
    prob_dict = {label: float(f"{probs[i]:.4f}") for i, label in enumerate(class_labels)}
    return pred_label, prob_dict, transform(img_pil)

# Grad-CAM generator
def generate_gradcam(image_tensor, model, class_idx):
    image_tensor = image_tensor.unsqueeze(0).to(device)
    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    target_layer = model.features[-1]
    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_full_backward_hook(backward_hook)

    output = model(image_tensor)
    loss = output[0, class_idx]
    model.zero_grad()
    loss.backward()

    grads = gradients[0].detach().cpu().numpy()[0]
    activs = activations[0].detach().cpu().numpy()[0]

    weights = np.mean(grads, axis=(1, 2))
    cam = np.zeros(activs.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * activs[i]
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam -= cam.min()
    cam /= cam.max()
    heatmap = np.uint8(cm.jet(cam)[..., :3] * 255)

    fh.remove()
    bh.remove()
    return heatmap

# PDF report generator
def generate_report(pred_label, gradcam_img):
    filename = f"Skin_Report_{int(time.time())}.pdf"
    path = os.path.join("/tmp", filename)
    c = canvas.Canvas(path, pagesize=letter)
    width, height = letter

    c.setFillColor(colors.HexColor("#003366"))
    c.rect(0, height - 80, width, 80, fill=True)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 22)
    c.drawString(50, height - 50, "MediScan Skin Diagnostics")

    c.setFont("Helvetica", 10)
    c.setFillColor(colors.black)
    c.drawString(50, height - 95, "📍 Patna AI Research Centre, Bihar, India")
    c.drawString(50, height - 110, "☎ +91-9876543210 | ✉ mediscanai@hospital.com")
    c.line(50, height - 115, width - 50, height - 115)

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 150, "AI-Powered Skin Disease Detection Report")

    c.setFont("Helvetica", 12)
    c.drawString(50, height - 180, "Prediction Result:")
    c.setFont("Helvetica-Bold", 14)
    c.drawString(220, height - 180, pred_label.upper())

    gradcam_path = os.path.join("/tmp", "gradcam_temp.jpg")
    Image.fromarray(gradcam_img).save(gradcam_path)
    c.setFont("Helvetica-Bold", 13)
    c.drawString(50, height - 210, "Grad-CAM Visualization:")
    try:
        c.drawImage(gradcam_path, 50, height - 460, width=200, height=200)
    except:
        c.drawString(50, height - 450, "Grad-CAM image not available.")

    recommendations = {
        "actinic_keratoses": {
            "Prevention": ["• Use broad-spectrum sunscreens.", "• Wear protective clothing."],
            "Tests": ["• Skin biopsy.", "• Dermatoscopy."],
            "Treatment": ["• Cryotherapy.", "• Topical 5-FU or imiquimod."]
        },
        "basal_cell_carcinoma": {
            "Prevention": ["• Avoid prolonged sun exposure.", "• Regular skin checks."],
            "Tests": ["• Skin biopsy.", "• Imaging for large tumors."],
            "Treatment": ["• Surgical excision.", "• Topical therapy."]
        },
        "benign_keratosis_like_lesions": {
            "Prevention": ["• Protect skin from UV.", "• Maintain hygiene."],
            "Tests": ["• Dermatoscopy.", "• Biopsy if uncertain."],
            "Treatment": ["• Observation.", "• Cryotherapy if needed."]
        },
        "dermatofibroma": {
            "Prevention": ["• Avoid skin injuries.", "• Prevent insect bites."],
            "Tests": ["• Clinical examination.", "• Excision if needed."],
            "Treatment": ["• Observation.", "• Surgical excision if problematic."]
        },
        "melanoma": {
            "Prevention": ["• Avoid peak UV hours.", "• Regular mole checks."],
            "Tests": ["• Skin biopsy.", "• Sentinel node biopsy.", "• Imaging if spread suspected."],
            "Treatment": ["• Wide local excision.", "• Immunotherapy for advanced cases."]
        },
        "nevus": {
            "Prevention": ["• Sun protection.", "• Self-examination."],
            "Tests": ["• Dermatoscopy.", "• Biopsy if atypical."],
            "Treatment": ["• Observation.", "• Excision if changing."]
        },
        "vascular_lesions": {
            "Prevention": ["• Avoid trauma.", "• Regular monitoring."],
            "Tests": ["• Clinical exam.", "• Imaging if deep lesion."],
            "Treatment": ["• Laser therapy.", "• Excision if needed."]
        }
    }

    y = height - 480
    rec = recommendations.get(pred_label.lower(), None)
    if rec:
        for section, items in rec.items():
            c.setFont("Helvetica-Bold", 13)
            c.drawString(270, y, f"{section}:")
            y -= 20
            c.setFont("Helvetica", 11)
            for line in items:
                c.drawString(270, y, line)
                y -= 15
            y -= 10

    c.setFont("Helvetica", 9)
    c.drawString(50, 50, f"Report Generated On: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    c.drawString(50, 35, "Generated by: MediScan AI System")
    c.drawString(50, 20, "Disclaimer: AI-generated report — consult a physician.")

    c.save()
    return path

# Final Gradio pipeline
def skin_disease_pipeline(img):
    try:
        print("🔍 Received image input")
        img_pil = Image.fromarray(img).convert("RGB")

        print("🔍 Running prediction")
        pred_label, prob_dict, image_tensor = predict_class(img_pil)
        print("✅ Prediction:", pred_label)

        print("🔍 Generating Grad-CAM")
        gradcam_img = generate_gradcam(image_tensor, model, class_labels.index(pred_label))
        print("✅ Grad-CAM generated")

        print("🔍 Generating PDF report")
        pdf_path = generate_report(pred_label, gradcam_img)
        print("✅ Report at:", pdf_path)

        max_prob = max(prob_dict.values())
        majority_result = f"{pred_label.upper()} ({max_prob*100:.2f}%)"

        return majority_result, prob_dict, gradcam_img, pdf_path

    except Exception as e:
        print("❌ Error:", e)
        traceback.print_exc()
        return "Error occurred", {"Error": str(e)}, np.zeros((224,224,3), dtype=np.uint8), None

# Gradio interface
iface = gr.Interface(
    fn=skin_disease_pipeline,
    inputs=gr.Image(type="numpy", label="Upload Skin Image"),
    outputs=[
        gr.Textbox(label="Predicted Disease & Probability"),
        gr.Label(label="Class Probabilities"),
        gr.Image(label="Grad-CAM Heatmap"),
        gr.File(label="Download PDF Report")
    ],
    title="Skin Disease Detector with AI Report",
    description="Upload a skin lesion image to classify the disease, see Grad-CAM visualization, and download a PDF report."
)

if __name__ == "__main__":
    iface.launch()
