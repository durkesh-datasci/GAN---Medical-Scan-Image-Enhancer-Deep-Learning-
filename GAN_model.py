import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.optim as optim
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Define Generator Network
class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)

# Define Discriminator Network
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)

# Function to load and preprocess images
def load_dataset(image_folder, transform, batch_size):
    dataset = datasets.ImageFolder(root=image_folder, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define image transformations
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load your dataset
image_folder = "C:\\Users\\Dinesh\\Downloads\\brain_tumor"
batch_size = 4
dataloader = load_dataset(image_folder, transform, batch_size)

# Initialize the generator and discriminator
latent_dim = 100
netG = Generator(latent_dim).to(device)
netG.apply(weights_init)
netD = Discriminator().to(device)
netD.apply(weights_init)

# Define the loss function and optimizers
criterion = nn.BCELoss()
optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Learning rate schedulers
schedulerG = optim.lr_scheduler.StepLR(optimizerG, step_size=30, gamma=0.1)
schedulerD = optim.lr_scheduler.StepLR(optimizerD, step_size=30, gamma=0.1)

# Training the GAN model
num_epochs = 50
fixed_noise = torch.randn(16, latent_dim, 1, 1, device=device)

for epoch in range(num_epochs):
    for i, data in enumerate(dataloader, 0):
        # Update Discriminator
        netD.zero_grad()
        real_images, _ = data
        real_images = real_images.to(device)

        output = netD(real_images).view(-1)
        label = torch.full((output.size(0),), 1, dtype=torch.float32, device=device)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        fake_images = netG(torch.randn(output.size(0), latent_dim, 1, 1, device=device))
        label.fill_(0)
        output = netD(fake_images.detach()).view(-1)
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()

        # Update Generator
        netG.zero_grad()
        label.fill_(1)
        output = netD(fake_images).view(-1)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()  # This is the correct place to define D_G_z2
        optimizerG.step()

        # Print training stats
        if i % 100 == 0:
            print(f"[Epoch {epoch+1}/{num_epochs}] [Batch {i}/{len(dataloader)}] "
                  f"[D Loss: {errD.item():.4f}] [G Loss: {errG.item():.4f}] "
                  f"[D(x): {D_x:.4f}] [D(G(z)): {D_G_z1:.4f}/{D_G_z2:.4f}]")

    # Step the learning rate scheduler
    schedulerG.step()
    schedulerD.step()

    # Generate and display images after each epoch
    with torch.no_grad():
        fake_images = netG(fixed_noise).detach().cpu()

    # Display generated images for the current epoch
    plt.figure(figsize=(10, 10))
    for idx in range(fake_images.size(0)):
        plt.subplot(4, 4, idx + 1)
        plt.imshow(fake_images[idx][0], cmap='gray')
        plt.axis('off')
    plt.show()

# Function to apply sharpening filter to an image
def sharpen_image(image):
    # Create the sharpening kernel
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    # Sharpen the image
    sharpened_image = cv2.filter2D(image, -1, kernel)
    return sharpened_image

# Function to load, resize, and process a set of images
def process_images(image_folder, target_size=(256, 256)):
    # List to store processed images
    processed_images = []

    # Iterate through each image in the folder
    for filename in os.listdir(image_folder):
        # Load the image
        image_path = os.path.join(image_folder, filename)
        image = cv2.imread(image_path)

        # Resize the image
        resized_image = cv2.resize(image, target_size)

        # Apply sharpening filter
        sharpened_image = sharpen_image(resized_image)

        # Convert to RGB and append to the list
        processed_images.append(cv2.cvtColor(sharpened_image, cv2.COLOR_BGR2RGB))

        # Break the loop after processing 16 images
        if len(processed_images) >= 16:
            break

    # Plot the processed images in a 4x4 grid
    plt.figure(figsize=(10, 10))
    for idx in range(len(processed_images)):
        plt.subplot(4, 4, idx + 1)
        plt.imshow(processed_images[idx])
        plt.axis('off')
    plt.show()

# Define the path to the folder containing images
image_folder = "C:\\Users\\Dinesh\\Downloads\\brain_tumor\\tumor"

# Process the images in the folder
process_images(image_folder)

#Classification Model


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from PIL import Image
import numpy as np
import os


import os
import random
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np
import tensorflow as tf
from PIL import Image

def split_dataset(image_dir, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "The sum of train, val and test ratios must be 1.0"

    classes = [d for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d))]
    for class_name in classes:
        class_path = os.path.join(image_dir, class_name)
        images = os.listdir(class_path)
        random.shuffle(images)
        
        train_count = int(len(images) * train_ratio)
        val_count = int(len(images) * val_ratio)
        
        train_images = images[:train_count]
        val_images = images[train_count:train_count + val_count]
        test_images = images[train_count + val_count:]
        
        for split_name, split_images in zip(['train', 'validation', 'test'], [train_images, val_images, test_images]):
            split_dir = os.path.join(output_dir, split_name, class_name)
            os.makedirs(split_dir, exist_ok=True)
            for image in split_images:
                shutil.copy(os.path.join(class_path, image), os.path.join(split_dir, image))

image_dir = r"C:\Users\Dinesh\Downloads\brain_tumor_dataset"
output_dir = r"C:\Users\Dinesh\Downloads\dataset"

split_dataset(image_dir, output_dir)

dataset_dir = r"C:\Users\Dinesh\Downloads\dataset"

train_datagen = ImageDataGenerator(rescale=0.255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=0.255)
test_datagen = ImageDataGenerator(rescale=0.255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(dataset_dir, 'train'),
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

validation_generator = val_datagen.flow_from_directory(
    os.path.join(dataset_dir, 'validation'),
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    os.path.join(dataset_dir, 'test'),
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Step 2: Model Creation
model = Sequential([
    tf.keras.Input(shape=(224, 224, 3)),  # Correct way to specify the input shape
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')  # Changed to 4 classes
])

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
# Example usage without .repeat() method
train_generator = train_datagen.flow_from_directory(
    os.path.join(dataset_dir, 'train'),
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Train the model with steps_per_epoch=None
model.fit(
    train_generator,
    steps_per_epoch=None,  # Allow the generator to loop indefinitely
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=20
)

# Save the model
model.save('tumor_classification_model.keras')

# Step 4: Model Evaluation
test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples // test_generator.batch_size)
print(f'Test accuracy: {test_acc}')

# Step 5: Image Classification
def preprocess_image(image_path, target_size=(224, 224)):
    image = Image.open(image_path)
    image = image.resize(target_size)
    image_array = np.array(image)
    image_array = image_array / 255.0  # Normalizing the image
    image_array = np.expand_dims(image_array, axis=0)  # Adding batch dimension
    return image_array

def classify_image(image_path, model_path):
    preprocessed_image = preprocess_image(image_path)
    model = tf.keras.models.load_model(model_path)
    predictions = model.predict(preprocessed_image)
    class_idx = np.argmax(predictions, axis=1)[0]
    
    class_labels = {
        0: "Healthy",
        1: "Pituitary",
        2: "Meningioma",
        3: "Glioma"
    }

    descriptions = {
        0: "No tumor present. The tissue appears healthy with no signs of abnormal growth or malignancy. Routine monitoring and regular check-ups are recommended to ensure continued health.",
        1: "Pituitary tumor detected. This tumor affects the pituitary gland, which regulates hormones. Immediate consultation with an endocrinologist or neurosurgeon is recommended to determine the appropriate treatment, which may include medication, surgery, or radiation.",
        2: "Meningioma detected. This tumor arises from the meninges, the protective membranes around the brain and spinal cord. Further medical evaluation is necessary to assess its size and impact. Treatment options include monitoring, surgical removal, or radiation therapy, depending on the tumor's characteristics.",
        3: "Glioma identified. Originating in the brain's glial cells, this tumor requires significant medical intervention. Treatment typically involves a combination of surgery, radiation, and chemotherapy. Prompt consultation with a neuro-oncologist is essential to develop a comprehensive treatment plan."
    }

    classification = class_labels[class_idx]
    description = descriptions[class_idx]
    
    return classification, description

# Example usage
image_path = r"C:\Users\Dinesh\Downloads\brain_tumor_dataset\meningioma\meningioma91.jpg"
model_path = 'tumor_classification_model.keras'
classification, description = classify_image(image_path, model_path)

import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# Tkinter GUI class
class Application:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Enhancement & Classification")
        
        self.original_image_label = tk.Label(root)
        self.original_image_label.pack(pady=10)
        
        self.enhanced_image_label = tk.Label(root)
        self.enhanced_image_label.pack(pady=10)
        
        self.classification_label = tk.Label(root)
        self.classification_label.pack(pady=10)
        
        self.upload_button = tk.Button(root, text="Upload Image", command=self.upload_image)
        self.upload_button.pack(pady=10)
        
    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            # Read the image using OpenCV
            original_image = cv2.imread(file_path)
            # Convert BGR (OpenCV format) to RGB (PIL format)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            # Convert to PIL image
            original_image_pil = Image.fromarray(original_image)
            # Convert the PIL image to a Tkinter-compatible image
            original_image_tk = ImageTk.PhotoImage(original_image_pil)

            # Display the original image
            self.original_image_label.config(image=original_image_tk)
            self.original_image_label.image = original_image_tk

            # Pass the image to the sharpen_image function
            sharpened_image_cv = sharpen_image(original_image)
            # Convert the sharpened OpenCV image to a PIL image
            sharpened_image_pil = Image.fromarray(sharpened_image_cv)
            # Convert the PIL image to a Tkinter-compatible image
            sharpened_image_tk = ImageTk.PhotoImage(sharpened_image_pil)

            # Display the sharpened image
            self.enhanced_image_label.config(image=sharpened_image_tk)
            self.enhanced_image_label.image = sharpened_image_tk
            
            # Classify image and display classification
            description = classify_image(file_path,r"C:\Users\Dinesh\Downloads\tumor_classification_model.keras")
            self.classification_label.config(text=f"Description: {description}")

# Create Tkinter application
root = tk.Tk()
app = Application(root)
root.mainloop()
