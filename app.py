import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from PIL import Image


DATA_DIR = "PokemonData"
MODEL_PATH = "resnet18_layer4_finetune.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def load_model():
    dataset = datasets.ImageFolder(DATA_DIR)
    class_names = dataset.classes
    num_classes = len(class_names)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    return model, class_names


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


def predict(image, model, class_names):
    image = image.convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        top5_prob, top5_idx = torch.topk(probabilities, 5)

    results = []
    for prob, idx in zip(top5_prob[0], top5_idx[0]):
        results.append({
            "Pokemon": class_names[idx.item()],
            "Probability": f"{prob.item() * 100:.2f}%"
        })

    return results


st.title("Pokemon Image Classifier")
st.write("Upload a Pokemon image and the model will predict its name.")

model, class_names = load_model()

uploaded_file = st.file_uploader(
    "Choose a Pokemon image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    results = predict(image, model, class_names)

    st.subheader("Top-5 Predictions")
    st.table(results)