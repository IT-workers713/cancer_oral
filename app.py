import os
import io
import base64
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

# لازم قبل import tensorflow باش يقلل اللوغات
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import cv2
from PIL import Image, ImageOps
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# -----------------------------
# Config
# -----------------------------
@dataclass(frozen=True)
class AppConfig:
    MODEL_PATH: str = "best_model_MobileNetV2.keras"
    IMG_SIZE: Tuple[int, int] = (224, 224)
    MAX_MB: int = 10
    # Threshold heatmap -> bbox
    HEATMAP_THRESHOLD: float = 0.5
    # Overlay weights
    OVERLAY_ALPHA: float = 0.6
    HEATMAP_ALPHA: float = 0.4
    # Allowed extensions
    ALLOWED_EXTS: Tuple[str, ...] = ("jpg", "jpeg", "png")


# -----------------------------
# Logging
# -----------------------------
def setup_logging(app: Flask) -> None:
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    app.logger.setLevel(level)

    # إذا ماكانش handlers (بعض البيئات)
    if not app.logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s in %(module)s: %(message)s"
        )
        handler.setFormatter(formatter)
        app.logger.addHandler(handler)


# -----------------------------
# ML Helpers (model + gradcam)
# -----------------------------
def load_model_once(model_path: str) -> tf.keras.Model:
    # compile=False غالباً أسرع وأأمن إذا راك غير inference
    return tf.keras.models.load_model(model_path, compile=False)


def get_last_conv_layer_name(model: tf.keras.Model) -> str:
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("Aucune couche Conv2D trouvée dans le modèle.")


def build_grad_model(model: tf.keras.Model, last_conv_layer_name: str) -> tf.keras.Model:
    return tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv_layer_name).output, model.output],
    )


def make_gradcam_heatmap(
    img_array: np.ndarray,
    grad_model: tf.keras.Model,
    target_class: str = "cancer",
) -> np.ndarray:
    """
    Sortie sigmoid = P(NON-CANCER) (label=1)
    Donc:
      - target_class='cancer'    => loss = 1 - p_non_cancer
      - target_class='non_cancer'=> loss = p_non_cancer
    """
    img_tensor = tf.convert_to_tensor(img_array)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)

        # بعض الموديلات ترجع list
        if isinstance(predictions, (list, tuple)):
            predictions = predictions[0]

        prob_non_cancer = predictions[:, 0]
        loss = (1.0 - prob_non_cancer) if target_class == "cancer" else prob_non_cancer

    grads = tape.gradient(loss, conv_outputs)
    # mean over spatial dims
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]  # (H,W,C)
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    heatmap = tf.maximum(heatmap, 0)
    denom = tf.reduce_max(heatmap) + tf.keras.backend.epsilon()
    heatmap = heatmap / denom

    return heatmap.numpy().astype(np.float32)


# -----------------------------
# Image Helpers
# -----------------------------
def allowed_file(filename: str, allowed_exts: Tuple[str, ...]) -> bool:
    if not filename or "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[-1].lower()
    return ext in set(allowed_exts)


def pil_to_base64(pil_img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/{fmt.lower()};base64,{b64}"


def bgr_to_base64(bgr_img: np.ndarray) -> str:
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    return pil_to_base64(pil_img, fmt="PNG")


def read_image_rgb(file_storage) -> Image.Image:
    """
    - يقرأ الصورة من stream
    - يصلّح orientation (EXIF)
    - يحولها لـ RGB
    """
    img = Image.open(file_storage.stream)
    img = ImageOps.exif_transpose(img)  # مهم لصور الهاتف
    return img.convert("RGB")


def preprocess_for_mobilenet(pil_img: Image.Image, img_size: Tuple[int, int]) -> np.ndarray:
    """
    Resize -> np.float32 -> preprocess_input -> (1,H,W,3)
    """
    img_resized = pil_img.resize(img_size)
    x = np.array(img_resized, dtype=np.float32)
    x = preprocess_input(x)
    x = np.expand_dims(x, axis=0)
    return x


# -----------------------------
# BBox extraction from heatmap
# -----------------------------
def get_bounding_box_from_heatmap(
    heatmap: np.ndarray,
    threshold: float = 0.5,
) -> Optional[Tuple[int, int, int, int]]:
    """
    heatmap: (H,W) float [0..1]
    returns: (x, y, w, h) of largest component, or None
    """
    # binarize
    binary = (heatmap >= threshold).astype(np.uint8) * 255

    # optional: close small holes (يساعد bbox يكون أنظف)
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 10:  # filter tiny noise
        return None

    x, y, w, h = cv2.boundingRect(largest)
    return x, y, w, h


# -----------------------------
# Flask App Factory
# -----------------------------
def create_app(config: AppConfig = AppConfig()) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = config.MAX_MB * 1024 * 1024
    setup_logging(app)

    # Load model once
    if not os.path.exists(config.MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {config.MODEL_PATH}")

    app.logger.info("Loading model: %s", config.MODEL_PATH)
    model = load_model_once(config.MODEL_PATH)
    last_conv = get_last_conv_layer_name(model)
    grad_model = build_grad_model(model, last_conv)
    app.logger.info("Model loaded. LAST_CONV_LAYER=%s", last_conv)

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.post("/")
    def predict():
        # 1) validate upload
        if "file" not in request.files:
            return render_template("index.html", error="Aucun fichier envoyé.")

        file = request.files["file"]
        if not file or file.filename == "":
            return render_template("index.html", error="Nom de fichier vide.")

        safe_name = secure_filename(file.filename)
        if not allowed_file(safe_name, config.ALLOWED_EXTS):
            return render_template(
                "index.html",
                error="Format invalide. Utilise JPG / JPEG / PNG.",
            )

        try:
            # 2) read & keep original
            img_pil = read_image_rgb(file)
            original_b64 = pil_to_base64(img_pil)

            # 3) preprocess
            x = preprocess_for_mobilenet(img_pil, config.IMG_SIZE)

            # 4) predict: sigmoid => P(non-cancer)
            pred = model.predict(x, verbose=0)
            raw_prob = float(pred[0][0])

            prob_non_cancer = raw_prob
            prob_cancer = 1.0 - raw_prob
            resultat = "CANCER" if raw_prob < 0.5 else "NON-CANCER"

            # 5) grad-cam for cancer
            heatmap = make_gradcam_heatmap(x, grad_model, target_class="cancer")

            # 6) overlay on resized image (224)
            img_resized = img_pil.resize(config.IMG_SIZE)
            img_cv = cv2.cvtColor(np.array(img_resized), cv2.COLOR_RGB2BGR)

            heatmap_resized = cv2.resize(heatmap, (img_cv.shape[1], img_cv.shape[0]))
            heatmap_color = cv2.applyColorMap(
                np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
            )

            superimposed = cv2.addWeighted(
                img_cv,
                config.OVERLAY_ALPHA,
                heatmap_color,
                config.HEATMAP_ALPHA,
                0,
            )

            bbox = get_bounding_box_from_heatmap(
                heatmap_resized, threshold=config.HEATMAP_THRESHOLD
            )
            if bbox is not None:
                x0, y0, w, h = bbox
                cv2.rectangle(
                    superimposed,
                    (x0, y0),
                    (x0 + w, y0 + h),
                    (0, 255, 0),
                    3,
                )

            overlay_b64 = bgr_to_base64(superimposed)

            return render_template(
                "index.html",
                original_image=original_b64,
                overlay_image=overlay_b64,
                resultat=resultat,
                prob_cancer=f"{prob_cancer*100:.2f}",
                prob_non_cancer=f"{prob_non_cancer*100:.2f}",
            )

        except Exception as e:
            app.logger.exception("Prediction failed: %s", str(e))
            return render_template(
                "index.html",
                error="Erreur lors du traitement de l'image. Vérifie le fichier et réessaie.",
            )

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "last_conv_layer": last_conv,
            "img_size": list(config.IMG_SIZE),
            "max_mb": config.MAX_MB,
        }

    return app


# -----------------------------
# Entry point
# -----------------------------
app = create_app()

if __name__ == "__main__":
    # Production: debug=0 (ولا تستعمل Flask dev server في prod)
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=debug)