from flask import Flask, request, render_template, send_file
from PIL import Image
import io
import torch
import torchvision.transforms as transforms
from style_transfer import *

app = Flask(__name__)

# Index route to render the HTML form
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle image processing
@app.route('/process', methods=['POST'])
def process_image():
    if 'file_content' not in request.files or 'file_style' not in request.files:
        return "Both content and style images are required."
    
    file_content = request.files['file_content']
    file_style = request.files['file_style']
    
    if file_content.filename == '' or file_style.filename == '':
        return "Both content and style images are required."
    
    content_image = Image.open(io.BytesIO(file_content.read()))
    style_image = Image.open(io.BytesIO(file_style.read()))

    # Convert images to tensors and run the style transfer function
    content_tensor = image_to_tensor(content_image)
    style_tensor = image_to_tensor(style_image)

    generated_image_pil = styler()

    # Save the generated image to a temporary file
    img_byte_array = io.BytesIO()
    generated_image_pil.save(img_byte_array, format='PNG')
    img_byte_array.seek(0)

    # Return the generated image as a response
    return send_file(img_byte_array, mimetype='image/png')

# Function to convert PIL image to PyTorch tensor
def image_to_tensor(image):
    loader = transforms.Compose([transforms.Resize((512,512)), transforms.ToTensor()])
    image = loader(image).unsqueeze(0)
    return image.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"), torch.float)

# Function to convert tensor to PIL image
def tensor_to_image(tensor):
    tensor = tensor.squeeze(0).detach().cpu()
    tensor = transforms.ToPILImage()(tensor)
    return tensor

if __name__ == '__main__':
    app.run(debug=True)
