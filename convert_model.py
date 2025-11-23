import tensorflow as tf
import os

models = [
    "main_cancer_model.keras",
    "all_model.keras",
    "bladder_cancer_model.keras",
    "brain_cancer_model.keras",
    "breast_cancer_model.keras",
    "cervikal_cancer_model.keras",
    "colon_cancer_model.keras",
    "esophageal_cancer_model.keras",
    "kidney_cancer_model.keras",
    "lymphoma_cancer_model.keras",
    "oral_cancer_model.keras",
    "ovarian_cancer_model.keras",
    "pancreatic_cancer_model.keras",
    "skin_cancer_model.keras",
    "thyroid_cancer_model.keras"
]

for model_name in models:
    if not os.path.exists(model_name):
        print(f"❌ Skipping missing model: {model_name}")
        continue

    print(f"🔄 Converting {model_name} ...")

    model = tf.keras.models.load_model(model_name)
    new_name = model_name.replace(".keras", "_tf2.keras")
    model.save(new_name, save_format="keras")

    print(f"✅ Saved: {new_name}")

print("🎉 All conversions done!")
