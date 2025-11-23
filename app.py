import gradio as gr
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import os

# Print TensorFlow version (SAFE)
print(f"TensorFlow version: {tf.__version__}")
# ❌ Removed keras.__version__ because it does NOT exist in TF 2.15


# --- Main model ---
main_model_path = "main_cancer_model.keras"
print(f"Loading main model from: {main_model_path}")

try:
    main_model = keras.models.load_model(main_model_path, compile=False)
    print("Main model loaded successfully.\n")
except Exception as e:
    print(f"Error loading model with Keras 2: {e}")
    print("Attempting alternative loading method...")
    main_model = keras.models.load_model(
        main_model_path,
        compile=False,
        custom_objects=None,
        safe_mode=False
    )
    print("Model loaded with fallback method.\n")

# --- Class names ---
class_names = [
    "ALL", "Bladder Cancer", "Brain Cancer", "Breast Cancer", "Cervical Cancer",
    "Esophageal Cancer", "Kidney Cancer", "Lung and Colon Cancer", "Lymphoma",
    "Oral Cancer", "Ovarian Cancer", "Pancreatic Cancer", "Skin Cancer", "Thyroid Cancer"
]

# --- Subclasses dictionary ---
subclasses = {
    "ALL": ["all_benign", "all_early", "all_pre", "all_pro"],
    "Bladder Cancer": ["bladder_muscle_invasive", "bladder_non_muscle_invasive"],
    "Brain Cancer": ["brain_glioma", "brain_menin", "brain_tumor"],
    "Breast Cancer": ["breast_benign", "breast_malignant"],
    "Cervical Cancer": ["cervix_dyk", "cervix_koc", "cervix_mep", "cervix_pab", "cervix_sfi"],
    "Esophageal Cancer": ["esophagus_benign", "esophagus_malignant"],
    "Kidney Cancer": ["kidney_normal", "kidney_tumor"],
    "Lung and Colon Cancer": ["colon_aca", "colon_bnt", "lung_aca", "lung_bnt", "lung_scc"],
    "Lymphoma": ["lymph_cll", "lymph_fl", "lymph_mcl"],
    "Oral Cancer": ["oral_normal", "oral_scc"],
    "Ovarian Cancer": [
        "ovarian_clear_cell_carcinoma", "ovarian_endometrioid",
        "ovarian_low_grade_serous", "ovarian_mucinous_carcinoma",
        "ovarian_high_grade_serous_carcinoma"
    ],
    "Pancreatic Cancer": ["pancreatic_normal", "pancreatic_tumor"],
    "Skin Cancer": [
        "Skin_Acne", "Skin_Actinic Keratosis", "Skin_Basal Cell Carcinoma",
        "Skin_Chickenpox", "Skin_Dermato Fibroma", "Skin_Dyshidrotic Eczema",
        "Skin_Melanoma", "Skin_Nail Fungus", "Skin_Nevus", "Skin_Normal Skin",
        "Skin_Pigmented Benign Keratosis", "Skin_Ringworm", "Skin_Seborrheic Keratosis",
        "Skin_Squamous Cell Carcinoma", "Skin_Vascular Lesion"
    ],
    "Thyroid Cancer": ["thyroid_benign", "thyroid_malignant"]
}

# --- Cache for loaded submodels ---
loaded_submodels = {}

def preprocess_image(img):
    img = img.convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def load_submodel(cancer_type):
    if cancer_type not in loaded_submodels:
        submodel_name = f"{cancer_type.lower().replace(' ', '_')}_model.keras"
        if os.path.exists(submodel_name):
            print(f"Loading submodel: {submodel_name}")
            loaded_submodels[cancer_type] = keras.models.load_model(submodel_name, compile=False)
        else:
            print(f"Submodel not found: {submodel_name}")
            loaded_submodels[cancer_type] = None
    return loaded_submodels[cancer_type]

def predict_image(img):
    try:
        img_array = preprocess_image(img)

        # Main prediction
        main_pred = main_model.predict(img_array, verbose=0)
        main_index = np.argmax(main_pred[0])
        cancer_type = class_names[main_index]
        main_confidence = float(main_pred[0][main_index])

        # Subclass prediction
        submodel = load_submodel(cancer_type)
        if submodel:
            sub_pred = submodel.predict(img_array, verbose=0)
            if len(sub_pred[0]) == len(subclasses[cancer_type]):
                sub_index = np.argmax(sub_pred[0])
                subclass_name = subclasses[cancer_type][sub_index]
                subclass_confidence = float(sub_pred[0][sub_index])
            else:
                subclass_name = "Model mismatch"
                subclass_confidence = 0.0
        else:
            subclass_name = "No submodel available"
            subclass_confidence = 0.0

        return {
            "cancer_type": cancer_type,
            "confidence": main_confidence,
            "subclass": subclass_name,
            "subclass_confidence": subclass_confidence
        }
    except Exception as e:
        print(f"Error during prediction: {e}")
        return {
            "cancer_type": "Error",
            "confidence": 0.0,
            "subclass": str(e),
            "subclass_confidence": 0.0
        }

def predict_gradio(img):
    if img is None:
        return "No image uploaded", "N/A", 0.0, 0.0
    
    result = predict_image(img)
    
    main_text = f"{result['cancer_type']} ({result['confidence']:.2%})"
    sub_text = f"{result['subclass']}" if result['subclass'] else "N/A"
    
    return main_text, sub_text, result['confidence'], result['subclass_confidence']

# --- Gradio Interface ---
demo = gr.Interface(
    fn=predict_gradio,
    inputs=gr.Image(type="pil", label="Upload Medical Image"),
    outputs=[
        gr.Text(label="🔬 Cancer Type & Confidence"),
        gr.Text(label="🧬 Subclass"),
        gr.Number(label="Main Confidence Score"),
        gr.Number(label="Subclass Confidence Score")
    ],
    title="🏥 Multi-Cancer Classification System",
    description=(
        "Upload a medical image to classify the cancer type and subclass. "
        "Supports 14 cancer types with detailed subclassification."
    ),
    allow_flagging="never"
)

if __name__ == "__main__":
    demo.launch()
