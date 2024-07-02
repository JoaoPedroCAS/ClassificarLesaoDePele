import os
import numpy as np
from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3, preprocess_input as preprocess_inception
from keras.applications.vgg19 import VGG19, preprocess_input as preprocess_vgg
from keras.applications.xception import Xception, preprocess_input as preprocess_xception

# Define paths
drive_path = 'C:\\Users\\jpedr\\OneDrive\\Documentos\\TCC\\Codigos\\CancerDePele\\cancer\\'
entrada = os.path.join(drive_path, 'data.txt')
dir_dataset = os.path.join(drive_path, 'data')
dir_destino = os.path.join(drive_path, 'libsvm')

# Create destination directory if it doesn't exist
os.makedirs(dir_destino, exist_ok=True)

# Define image dimensions
img_rows, img_cols = 224, 224

# Read input file
with open(entrada, 'r') as arq:
    conteudo_entrada = arq.readlines()

def process_images(model, preprocess_input, output_file, data, batch_size=32):
    """
    Process images and extract features.

    Parameters:
    model : The pre-trained model to use for feature extraction.
    preprocess_input : The preprocessing function to use on the data.
    output_file : The file to write the extracted features to.
    data : The data to process.
    batch_size : The number of images to process at a time.
    """
    with open(output_file, 'w') as file:
        for i in range(0, len(data), batch_size):
            batch_data = data[i:i+batch_size]
            batch_images = []
            for line in batch_data:
                nome, classe = line.split()
                img_path = os.path.join(dir_dataset, nome)
                img = image.load_img(img_path, target_size=(img_rows, img_cols))
                img_data = image.img_to_array(img)
                img_data = np.expand_dims(img_data, axis=0)
                img_data = preprocess_input(img_data)
                batch_images.append(img_data)
            batch_images = np.vstack(batch_images)
            features = model.predict(batch_images)
            for j, features_np in enumerate(features):
                features_np = features_np.flatten()
                features_str = " ".join(f"{k+1}:{features_np[k]}" for k in range(features_np.size))
                file.write(f"{batch_data[j].split()[1]} {features_str}\n")


model = Xception(weights="imagenet", include_top=False)
process_images(model, preprocess_inception, os.path.join(dir_destino, 'data_Xception.txt'), conteudo_entrada)
