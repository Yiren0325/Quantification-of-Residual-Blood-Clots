import torch
import os
import config
from torchvision import transforms
from PIL import Image



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
img_name = 'sample_data_30.png'
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
print(f'predicted class: {config.dataSetting["idx_to_class"][top_indices[0][0].item()]}% ({top_probs[0][0]:6.4f})')