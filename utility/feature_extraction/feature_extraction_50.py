import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import scipy.io as sio
from tqdm import tqdm
import random
import torch.utils.data
import torch.optim as optim

random.seed(1)
torch.manual_seed(1)
np.random.seed(1)


class ImageDatasetWithLabelsFromFilename(torch.utils.data.Dataset):

    def __init__(self, image_dir, filename_delimiter, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.delimiter = filename_delimiter
        
        supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
        image_files_list = sorted([
            f for f in os.listdir(self.image_dir) 
            if os.path.splitext(f)[1].lower() in supported_formats
        ])
        
        random.shuffle(image_files_list)
        
        self.image_files = image_files_list

        all_class_names = sorted(list(set([f.split(self.delimiter)[0] for f in self.image_files])))
        self.class_to_idx = {class_name: i for i, class_name in enumerate(all_class_names)}

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_name)
        
        class_name = image_name.split(self.delimiter)[0]
        label = self.class_to_idx[class_name]
        
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (224, 224), (0, 0, 0)) # Fallback
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label).long(), image_path



def extract_features_and_save(dataset):


    DATASET_IMG_PATH = f'data/{dataset}/img'  
    OUTPUT_DIR = f'Feature-Generation-datasets/{dataset}'   
    FILENAME_DELIMITER = '-' 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    weights = models.ResNet50_Weights.DEFAULT
    preprocess = weights.transforms(antialias=True)
    feature_extractor = models.resnet50(weights=weights)

    for param in feature_extractor.parameters():
        param.requires_grad = False
    
    feature_extractor = nn.Sequential(*list(feature_extractor.children())[:-1])
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()

    dataset_obj = ImageDatasetWithLabelsFromFilename(
        image_dir=DATASET_IMG_PATH,
        filename_delimiter=FILENAME_DELIMITER,
        transform=preprocess
    )
    dataloader = torch.utils.data.DataLoader(
        dataset_obj,
        batch_size=10, 
        shuffle=False,
        num_workers=2
    )


    print("Starting feature extraction from a single directory...")
    img_files = []
    features = []
    labels = []


    with torch.no_grad():
        for i, (batch_input, batch_target, impath) in enumerate(tqdm(dataloader, desc=f"{dataset}")):
            
            input_v = batch_input.to(device)
            output = feature_extractor(input_v)
            output = torch.flatten(output, 1).cpu()

            for j in range(len(batch_target)):

                img_files.append([np.array([impath[j]])])
                

                labels.append(np.array([batch_target[j].item() + 1], dtype=np.int16))
                

                features.append(output[j].numpy())

    print("Feature extraction complete.")


    if not features:
        print("No features were extracted. Exiting.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    output_filepath = os.path.join(OUTPUT_DIR, 'pretrained_resnet50.mat')

    sio.savemat(
        output_filepath, 
        mdict={
            'image_files': np.array(img_files, dtype=object),
            'features': np.array(features), 
            'labels': np.array(labels)
        }
    )

def finetune_model(dataset, epoch_num):

    DATASET_IMG_PATH = f'data/{dataset}/img' 
    OUTPUT_DIR = f'Feature-Generation-datasets/{dataset}'  
    FILENAME_DELIMITER = '-' 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset_obj = ImageDatasetWithLabelsFromFilename(DATASET_IMG_PATH, FILENAME_DELIMITER)
    num_classes = len(dataset_obj.class_to_idx)
    
    weights = models.ResNet50_Weights.DEFAULT
    train_transform = weights.transforms(antialias=True) 
    dataset_obj.transform = train_transform
    
    train_loader = torch.utils.data.DataLoader(dataset_obj, batch_size=10, shuffle=True, num_workers=2)
    num_classes = len(dataset_obj.class_to_idx)

    model = models.resnet50(weights=weights)
    eval_transform = weights.transforms(antialias=True)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    model.to(device)


    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epoch_num):
        model.train()
        running_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epoch_num}")
        for batch_input, batch_target, _ in progress_bar:
            batch_input = batch_input.to(device)
            batch_target = batch_target.to(device)
            
            optimizer.zero_grad()
            output = model(batch_input)
            loss = criterion(output, batch_target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
            
        epoch_loss = running_loss / len(train_loader)


    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, f"res50_finetuned_{epoch_num}.pth")
    torch.save(model.state_dict(), save_path)


    model.eval() 
    model.fc = nn.Identity() 

    dataset_obj.transform = eval_transform
    extract_loader = torch.utils.data.DataLoader(dataset_obj, batch_size=10, shuffle=False, num_workers=2)

    img_files, features, labels = [], [], []
    with torch.no_grad():
        for batch_input, batch_target, impath in tqdm(extract_loader):
            output = model(batch_input.to(device))
            output = torch.flatten(output, 1).cpu()
            for j in range(len(batch_target)):
                img_files.append([np.array([impath[j]])])
                labels.append(np.array([batch_target[j].item() + 1], dtype=np.int16))
                features.append(output[j].numpy())
                
    output_filepath = os.path.join(OUTPUT_DIR, f'resnet50_finetuned_{epoch_num}.mat')


    sio.savemat(
        output_filepath, 
        mdict={'image_files': np.array(img_files, dtype=object), 'features': np.array(features), 'labels': np.array(labels)}
    )

if __name__ == '__main__':
    dataset_list = ['SD','Road']
    # dataset_list = ['SD']
    epoch_list = [5]
    for dataset in dataset_list:
        extract_features_and_save(dataset)
        for epoch in epoch_list:
            finetune_model(dataset, epoch)