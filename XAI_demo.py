import torch
import os
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import slic


# Load the model
model = torch.load('model_set.pth')
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load and preprocess the image
data_dir = './sample_data'
img_name = 'sample_data_10.png'
img_path = os.path.join(data_dir, img_name)
img = Image.open(img_path)
input_tensor = transform(img).unsqueeze(0)

# Use GPU acceleration if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
input_tensor = input_tensor.to(device)

# Make predictions
output = model(input_tensor)
probs = torch.nn.functional.softmax(output, dim=1)
top_probs, top_indices = torch.topk(probs, 2)  # Get the top 2 most likely classes

# Print the most likely class and its probability
print(f"Top predictions: {top_indices[0]}")
print(f"Top probabilities: {top_probs[0]}")


# LIME needs to handle data in NumPy format
def batch_predict(images):
    model.eval()
    images = torch.stack([transform(Image.fromarray(img)).to(device) for img in images], dim=0)
    with torch.no_grad():
        preds = model(images)
    return torch.nn.functional.softmax(preds, dim=1).cpu().numpy()


# Initialize LIME image explainer
explainer = lime_image.LimeImageExplainer()

# Custom segmentation function for LIME
def custom_segmentation(image):
    return slic(image, n_segments=100, compactness=10, sigma=1)

# Explain the model's prediction for the image
explanation = explainer.explain_instance(np.array(img), 
                                        batch_predict,  # Prediction function
                                        top_labels=2,    # Explain the top 2 most likely classes
                                        hide_color=0,    # 隱ored pixel color
                                        num_samples=1000,  # Number of perturbed samples
                                        segmentation_fn=custom_segmentation)  # Custom segmentation function

# Get the most likely class indices
label_1 = explanation.top_labels[0]
label_2 = explanation.top_labels[1]
label_description = {0:'low clotting',1:'moderate clotting'}

# Get the images and masks for the top labels
temp_1, mask_1 = explanation.get_image_and_mask(label=label_1,
                                                positive_only=True,    # Show positive and negative features
                                                num_features=20,       # Show the top 10 features
                                                hide_rest=False)       # Do not hide other parts

temp_2, mask_2 = explanation.get_image_and_mask(label=label_2,
                                                positive_only=True,    # Show positive and negative features
                                                num_features=20,       # Show the top 10 features
                                                hide_rest=False)       # Do not hide other parts

# Convert masks to boolean arrays
mask_1 = mask_1 > 0
mask_2 = mask_2 > 0

# Function to apply color mask to the image
def apply_color_mask(image, mask, color, alpha=0.5):
    colored_image = image.copy()
    for c in range(3):  # Apply color to each channel
        colored_image[:, :, c] = np.where(mask, 
                                        colored_image[:, :, c] * (1 - alpha) + color[c] * alpha, 
                                        colored_image[:, :, c])
    return colored_image

# Apply color to the positive feature areas and adjust opacity
colored_temp_1 = apply_color_mask(temp_1, mask_1, [0, 255, 0], alpha=0.5)
colored_temp_2 = apply_color_mask(temp_2, mask_2, [0, 255, 0], alpha=0.5)

# Visualize the explanation results for the two classes
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Display the original image
axes[0].imshow(np.array(img) / 255.0)
axes[0].set_title("Original Image")
axes[0].axis('off')
# Display the result and predicted probability for class 1
axes[1].imshow(colored_temp_1 / 255.0)
axes[1].set_title(f"Predicted Probability of Class {label_1}: {top_probs[0][0]:.3f}")
axes[1].axis('off')

# Display the result and predicted probability for class 2
axes[2].imshow(colored_temp_2 / 255.0)
axes[2].set_title(f"Predicted Probability of Class {label_2}: {top_probs[0][1]:.3f}")
axes[2].axis('off')

# plt.tight_layout()
plt.savefig(f"./sample_data/{img_name}_xai.png")  # Save the resulting image
plt.close()
