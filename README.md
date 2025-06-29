# 🩺 MediScan AI — Skin Disease Detection with PDF Report & Grad-CAM

An AI-powered diagnostic tool that detects seven types of skin diseases from image input using a deep learning model. The app visualizes disease regions with Grad-CAM and generates a downloadable PDF report containing diagnosis, recommendations, and clinical information.

---

## 🚀 Features

- ✅ Skin disease classification (7 classes)
- ✅ Grad-CAM heatmap visualization
- ✅ Auto-generated PDF report with diagnosis, test suggestions, prevention & treatment
- ✅ Clean Gradio interface for interaction
- ✅ Deployable on Hugging Face or locally

---

## 🧠 Model Info

- **Architecture**: EfficientNet-B0
- **Framework**: PyTorch
- **Trained On**: Skin lesion images
- **Classes**:
  - `actinic_keratoses`
  - `basal_cell_carcinoma`
  - `benign_keratosis_like_lesions`
  - `dermatofibroma`
  - `melanoma`
  - `nevus`
  - `vascular_lesions`

---

## 💻 How to Run

1. Clone the repo:
    ```bash
    git clone https://github.com/your-username/mediscan-ai.git
    cd mediscan-ai
    ```

2. Install dependencies:
    ```bash
    pip install torch torchvision gradio numpy pillow opencv-python matplotlib reportlab
    ```

3. Place your trained model:
    - Put `best_skin_model.pth` in the root directory

4. Launch the app:
    ```bash
    python app.py
    ```

---

## 📄 Output

- **Predicted Class**: e.g., `Melanoma (92.87%)`
- **Grad-CAM**: Highlights region of concern
- **PDF Report**: Downloadable report with diagnosis and care suggestions

---


