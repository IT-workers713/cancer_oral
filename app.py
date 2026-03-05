import os
import io
import base64
import numpy as np
import cv2
from PIL import Image
from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# -----------------------------
# Config
# -----------------------------
MODEL_PATH = "best_model_MobileNetV2.keras"
IMG_SIZE = (224, 224)
MAX_MB = 10

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_MB * 1024 * 1024

# Optionnel: réduire le bruit des logs TF
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# -----------------------------
# Load model once (global)
# -----------------------------
def load_my_model():
    return tf.keras.models.load_model(MODEL_PATH)

def get_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("Aucune couche Conv2D trouvée dans le modèle.")

model = load_my_model()
LAST_CONV_LAYER = get_last_conv_layer(model)


# -----------------------------
# Grad-CAM (cible cancer / non-cancer)
# -----------------------------
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, target_class="cancer"):
    """
    Sortie sigmoid = P(NON-CANCER) (label=1)
    Donc:
      - target_class='cancer'    => loss = 1 - p_non_cancer
      - target_class='non_cancer'=> loss = p_non_cancer
    """
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)

        if isinstance(predictions, list):
            predictions = predictions[0]

        prob_non_cancer = predictions[:, 0]

        if target_class == "cancer":
            loss = 1.0 - prob_non_cancer
        else:
            loss = prob_non_cancer

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + tf.keras.backend.epsilon()
    return heatmap.numpy()


# -----------------------------
# BBox extraction from heatmap
# -----------------------------
def get_bounding_box_from_heatmap(heatmap, threshold=0.5):
    """
    heatmap: (H,W) float [0..1]
    """
    binary = (heatmap > threshold).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    return x, y, w, h


# -----------------------------
# Helpers
# -----------------------------
def pil_to_base64(pil_img, fmt="PNG"):
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/{fmt.lower()};base64,{b64}"

def bgr_to_base64(bgr_img):
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    return pil_to_base64(pil_img, fmt="PNG")

def allowed_file(filename):
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    return ext in {"jpg", "jpeg", "png"}


# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def index():
    return render_template("index.html")

@app.post("/")
def predict():
    if "file" not in request.files:
        return render_template("index.html", error="Aucun fichier envoyé.")

    file = request.files["file"]
    if file.filename == "":
        return render_template("index.html", error="Nom de fichier vide.")
    if not allowed_file(file.filename):
        return render_template("index.html", error="Format invalide. Utilise JPG / JPEG / PNG.")

    # Lire image
    img_pil = Image.open(file.stream).convert("RGB")
    original_b64 = pil_to_base64(img_pil)

    # Resize 224
    img_resized = img_pil.resize(IMG_SIZE)

    # Preprocess MobileNetV2 [-1,1]
    x = np.array(img_resized).astype(np.float32)
    x = preprocess_input(x)
    x = np.expand_dims(x, axis=0)

    # Predict: sigmoid => P(non-cancer)
    raw_prob = float(model.predict(x, verbose=0)[0][0])

    prob_cancer = 1.0 - raw_prob
    prob_non_cancer = raw_prob
    resultat = "CANCER" if raw_prob < 0.5 else "NON-CANCER"

    # Grad-CAM pour la classe cancer (comme ton Streamlit)
    heatmap = make_gradcam_heatmap(x, model, LAST_CONV_LAYER, target_class="cancer")

    # OpenCV overlay sur image 224
    img_cv = cv2.cvtColor(np.array(img_resized), cv2.COLOR_RGB2BGR)
    heatmap_resized = cv2.resize(heatmap, (img_cv.shape[1], img_cv.shape[0]))

    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(img_cv, 0.6, heatmap_color, 0.4, 0)

    bbox = get_bounding_box_from_heatmap(heatmap_resized, threshold=0.5)
    if bbox is not None:
        x0, y0, w, h = bbox
        cv2.rectangle(superimposed, (x0, y0), (x0 + w, y0 + h), (0, 255, 0), 3)

    overlay_b64 = bgr_to_base64(superimposed)

    return render_template(
        "index.html",
        original_image=original_b64,
        overlay_image=overlay_b64,
        resultat=resultat,
        prob_cancer=f"{prob_cancer*100:.2f}",
        prob_non_cancer=f"{prob_non_cancer*100:.2f}"
    )

@app.get("/health")
def health():
    return {"status": "ok", "last_conv_layer": LAST_CONV_LAYER}


if __name__ == "__main__":
    # Debug True فقط أثناء التطوير
    app.run(host="0.0.0.0", port=5000, debug=True)