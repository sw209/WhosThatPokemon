import matplotlib.pyplot as plt
import os

os.makedirs("images", exist_ok=True)

epochs = list(range(1, 11))

results = {
    "Custom CNN": {
        "train": [2.70, 7.65, 9.78, 13.01, 13.89, 16.65, 18.18, 20.00, 21.24, 23.77],
        "val":   [3.42, 9.38, 12.12, 15.05, 17.69, 16.03, 19.35, 24.24, 22.19, 25.71],
    },
    "ResNet18 FC only": {
        "train": [22.69, 68.48, 83.45, 88.98, 91.66, 93.55, 95.27, 96.54, 97.51, 98.07],
        "val":   [46.04, 66.47, 70.58, 73.80, 75.17, 78.20, 78.59, 78.30, 79.18, 78.30],
    },
    "ResNet18 layer4": {
        "train": [35.80, 87.75, 97.17, 99.18, 99.60, 99.71, 99.81, 99.81, 99.81, 99.77],
        "val":   [65.59, 84.65, 90.03, 91.89, 91.98, 92.57, 92.38, 92.38, 92.57, 92.57],
    },
    "MobileNetV2": {
        "train": [30.00, 76.35, 86.53, 92.04, 93.78, 95.48, 96.71, 97.32, 98.22, 98.24],
        "val":   [52.79, 66.18, 72.24, 73.22, 75.07, 75.95, 76.15, 76.44, 77.91, 78.20],
    },
}

# 1. Validation Accuracy 비교 그래프
plt.figure(figsize=(10, 6))

for model_name, data in results.items():
    plt.plot(epochs, data["val"], marker="o", label=model_name)

plt.title("Validation Accuracy Comparison")
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy (%)")
plt.xticks(epochs)
plt.ylim(0, 100)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("images/validation_accuracy_comparison.png", dpi=300)
plt.close()


# 2. Train Accuracy 비교 그래프
plt.figure(figsize=(10, 6))

for model_name, data in results.items():
    plt.plot(epochs, data["train"], marker="o", label=model_name)

plt.title("Training Accuracy Comparison")
plt.xlabel("Epoch")
plt.ylabel("Training Accuracy (%)")
plt.xticks(epochs)
plt.ylim(0, 100)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("images/training_accuracy_comparison.png", dpi=300)
plt.close()


# 3. 모델별 Train / Validation 그래프
for model_name, data in results.items():
    safe_name = model_name.lower().replace(" ", "_").replace("/", "_")

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, data["train"], marker="o", label="Train Accuracy")
    plt.plot(epochs, data["val"], marker="o", label="Validation Accuracy")

    plt.title(f"{model_name} Learning Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.xticks(epochs)
    plt.ylim(0, 100)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(f"images/{safe_name}_learning_curve.png", dpi=300)
    plt.close()

print("Learning curve images saved in images/ folder.")