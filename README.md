# GAN - Medical-Scan-Image-Enhancer-Deep-Learning-
A Deep Learning Model - GAN, to enhance the medical scan image to high quality image.
# Brain Tumor Detection and Classification

This project utilizes a Generative Adversarial Network (GAN) and a Convolutional Neural Network (CNN) for the detection and classification of brain tumors. It also includes a Tkinter-based GUI for image enhancement and classification.

## 📋 Overview

The project is divided into the following key components:

1. **Data Preparation**: Preprocessing and splitting the brain tumor dataset.
2. **GAN Model**: A Generative Adversarial Network for generating synthetic brain tumor images.
3. **CNN Classification Model**: A Convolutional Neural Network for classifying brain tumors into four categories.
4. **Image Enhancement**: A function to sharpen images using OpenCV.
5. **GUI Application**: A Tkinter-based interface for uploading images, enhancing them, and classifying them.

## 🚀 Installation

### Prerequisites

Ensure you have the following installed:

- Python 3.8 or higher
- PyTorch
- TensorFlow
- OpenCV
- Tkinter
- Matplotlib
- PIL (Pillow)

### Setup

1. **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/brain-tumor-detection.git
    cd brain-tumor-detection
    ```

2. **Install the required packages:**

    ```bash
    pip install -r requirements.txt
    ```

## 📂 Dataset

The dataset used in this project consists of brain tumor images categorized into four classes:

- **Healthy**
- **Pituitary**
- **Meningioma**
- **Glioma**

### 📁 Dataset Directory Structure

Ensure your dataset is organized as follows:

brain_tumor_dataset/
├── healthy/
├── pituitary/
├── meningioma/
└── glioma/

python
Copy code

## 🧠 Model Training

### 🖼️ Generative Adversarial Network (GAN)

The GAN model is used to generate synthetic brain tumor images. It consists of a **Generator** and a **Discriminator**:

```python
class Generator(nn.Module):
    # Code for the generator

class Discriminator(nn.Module):
    # Code for the discriminator
🧑‍⚕️ Convolutional Neural Network (CNN)
The CNN is used for classifying the brain tumor images into the four categories mentioned above:

python
Copy code
model = Sequential([
    # CNN architecture
])
🔧 Training the Models
Run the following script to train the models:

bash
Copy code
python train_models.py
🛠️ Image Enhancement
The sharpen_image function enhances the images by applying a sharpening filter using OpenCV:

python
Copy code
def sharpen_image(image):
    # Code for image sharpening
🖥️ GUI Application
A Tkinter-based GUI is provided for uploading images, enhancing them, and classifying them:

python
Copy code
root = tk.Tk()
app = Application(root)
root.mainloop()
🖼️ How to Use the GUI
Click "Upload Image" to select an image.
View the original and enhanced images.
The model will classify the image and provide a detailed description.
📈 Results
The model achieves a high accuracy on the test dataset. The classification model is capable of distinguishing between healthy tissue and various types of tumors with considerable precision.

🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

🧑‍💻 Author
Dinesh Babu - Initial work - GitHub
