import argparse
import os
import sys
from torchvision import datasets, transforms
from PIL import Image
import shutil
import json
import random
from sklearn.model_selection import train_test_split
sys.path.append(os.getcwd())

def save_to_jsonl(data, file_path):
    with open(file_path, 'w') as jsonl_file:
        for entry in data:
            jsonl_file.write(json.dumps(entry) + '\n')

def main(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

        transform = transforms.Compose([transforms.ToTensor()])
        mnist_dataset = datasets.MNIST(root = args.data_dir, train=True, download=True, transform=transform)
        for i in range(10):
            label_dir = os.path.join(args.output_dir, str(i))
            if not os.path.exists(label_dir):
                os.makedirs(label_dir)

        for idx, (image, label) in enumerate(mnist_dataset):
            image_pil = transforms.ToPILImage()(image)
            label_dir = os.path.join(args.output_dir, str(label))
            image_path = os.path.join(label_dir, f'{idx}.png')
            image_pil.save(image_path)
            if idx % 1000 == 0:
                print(f"Processed {idx} images")
    
    if not os.path.exists(args.train_dir):
        os.makedirs(args.train_dir)
    all_data = []
    for label in range(10): 
        each_label_max_data_len = 500
        label_dir = os.path.join(args.output_dir, str(label))
        if os.path.exists(label_dir):
            for img_name in os.listdir(label_dir):
                img_path = os.path.join(label_dir, img_name)
                if os.path.isfile(img_path):
                    all_data.append({"image_path": img_path, "label": label})
                    each_label_max_data_len = each_label_max_data_len - 1
                    if(each_label_max_data_len == 0): 
                        break
                    
    train_data, temp_data = train_test_split(all_data, test_size=0.3, random_state=42)  
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42) 
    save_to_jsonl(train_data, args.train_dir + "train.jsonl")
    save_to_jsonl(val_data, args.train_dir + "val.jsonl")
    save_to_jsonl(test_data, args.train_dir + "test.jsonl")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",type = str,default="datasets/MNIST/MNIST_data/")
    parser.add_argument("--output_dir",type = str,default = "datasets/MNIST/images/")
    parser.add_argument("--train_dir",type = str,default = "mnist_train/")
    args = parser.parse_args()
    main(args)
