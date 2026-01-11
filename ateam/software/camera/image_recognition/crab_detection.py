"""Detects crabs in images"""

import shutil
import time
import json
import tkinter as tk
from tkinter.filedialog import askopenfilename, askdirectory
import matplotlib.pyplot as plt
import cv2
import os
import torch
import torchvision
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from pycocotools.coco import COCO
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from engine import train_one_epoch, evaluate
from torchvision import models, transforms

try:
    import intel_extension_for_pytorch as ipex
except ImportError:
    ipex = None

# Returns a simple transform that converts a PIL image to a PyTorch tensor
def get_transform():
    return ToTensor()

class CocoDetectionDataset(Dataset):
    # Init function: loads annotation file and prepares list of image IDs
    def __init__(self, image_dir, annotation_path, transforms=None):
        self.image_dir = image_dir
        self.coco = COCO(annotation_path)
        self.image_ids = list(self.coco.imgs.keys())
        self.transforms = transforms

    # Returns total number of images
    def __len__(self):
        return len(self.image_ids)

    # Fetches a single image and its annotations
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.image_dir, os.path.basename(image_info['file_name']))
        image = Image.open(image_path).convert("RGB")

        # Load all annotations for this image
        annotation_ids = self.coco.getAnnIds(imgIds=image_id)
        annotations = self.coco.loadAnns(annotation_ids)

        # Extract bounding boxes and labels from annotations
        boxes = []
        labels = []
        for obj in annotations:
            xmin, ymin, width, height = obj['bbox']
            xmax = xmin + width
            ymax = ymin + height
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(obj['category_id'])

        # Convert annotations to PyTorch tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        area = torch.as_tensor([obj['area'] for obj in annotations], dtype=torch.float32)
        iscrowd = torch.as_tensor([obj.get('iscrowd', 0) for obj in annotations], dtype=torch.int64)

        # Package everything into a target dictionary
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd
        }

        # Apply transforms if any were passed
        if self.transforms:
            image = self.transforms(image)

        return image, target


class CrabDetector:
    def __init__(self):
        self.BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        
        # Setup Device
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            self.device = torch.device("xpu")
            print(f"Using Intel Arc GPU: {torch.xpu.get_device_name(0)}")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Using NVIDIA GPU (CUDA)")
        else:
            self.device = torch.device("cpu")
            print("Using CPU")

        # Class names
        self.label_list = ["", "European Green", "Native Rock", "Native Jonah"]
        
        # Configuration
        self.train_dir = os.path.join(self.BASE_DIR, "dataset_images/train")
        self.annot_path = os.path.join(self.BASE_DIR, "dataset_images/train/_annotations.coco.json")
        self.model_save_path = os.path.join(self.BASE_DIR, "faster-rcnn-torch")
        self.num_epochs = 5 # Default global setting from original file
        
        # Placeholders
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.optimizer = None

    def _get_dataset(self):
         return CocoDetectionDataset(
            image_dir=self.train_dir,
            annotation_path=self.annot_path,
            transforms=get_transform()
        )

    def prepare_training_data(self):
        # Load training dataset
        train_dataset = self._get_dataset()
        
        # Load validation dataset (same as train in original code)
        val_dataset = self._get_dataset()

        # DataLoaders
        self.train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
        self.val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
        
        return train_dataset # Return for class count usage

    def setup_training_model(self):
        # Load a pre-trained Faster R-CNN model
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        
        # Get dataset to count classes
        # Note: Original code did this by loading dataset first. 
        # Ideally we should call prepare_training_data first or just load the coco object.
        coco = COCO(self.annot_path)
        num_classes = len(coco.getCatIds()) + 1  # +1 for background class

        # Get the number of input features for the classifier head
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        # Replace the classifier head
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # Move to device
        self.model.to(self.device)

        # Optimizer
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

        if ipex and self.device.type == "xpu":
            print("Optimizing model and optimizer with Intel Extension for PyTorch (IPEX)")
            self.model, self.optimizer = ipex.optimize(self.model, optimizer=self.optimizer)

    def train(self, epochs=10):
        # Ensure model and data are ready
        if not self.model:
            self.setup_training_model()
        if not self.train_loader:
            self.prepare_training_data()
            
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            train_one_epoch(self.model, self.optimizer, self.train_loader, self.device, epoch, print_freq=25)
            evaluate(self.model, self.val_loader, device=self.device)
            
            if not os.path.exists(self.model_save_path):
                os.makedirs(self.model_save_path)
            torch.save(self.model.state_dict(), os.path.join(self.model_save_path, f"model_epoch_{epoch + 1}.pth"))

    def detect(self, image_input, threshold=0.8):
        """
        Runs detection on an image.
        image_input: path to image string
        Returns: list of dicts with keys 'label', 'score', 'box'
        """
        # Model setup for inference (if not already loaded or if converting from training)
        # Original code re-initializes model for test_mode.
        # We will assume 5 classes for inference based on original 'test_mode' logic.
        num_classes = 5 
        
        inference_model = models.detection.fasterrcnn_resnet50_fpn(pretrained=False, num_classes=num_classes)
        
        # Load weights
        # Uses self.num_epochs (which is 5 by default) to find the model
        model_path = os.path.join(self.model_save_path, f"model_epoch_{self.num_epochs}.pth")
        if not os.path.exists(model_path):
            print(f"Model path not found: {model_path}")
            return []
            
        inference_model.load_state_dict(torch.load(model_path))
        inference_model.to(self.device)
        inference_model.eval()

        # Handle Input
        image_bgr = None
        if isinstance(image_input, str):
            image_bgr = cv2.imread(image_input)
        else:
            # Assume numpy array (BGR)
            image_bgr = image_input

        if image_bgr is None:
            print("Could not load image.")
            return []

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)

        # Transform
        transform = transforms.Compose([transforms.ToTensor()])
        image_tensor = transform(image_pil).unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            predictions = inference_model(image_tensor)

        # Process results
        boxes_out = predictions[0]['boxes']
        labels_out = predictions[0]['labels']
        scores_out = predictions[0]['scores']

        results = []
        for i in range(len(boxes_out)):
            if scores_out[i] > threshold:
                box = boxes_out[i].cpu().numpy().astype(int)
                label_idx = labels_out[i]
                # Safe label access
                label_name = self.label_list[label_idx] if label_idx < len(self.label_list) else str(label_idx)
                score = scores_out[i].item()
                
                results.append({
                    "label": label_name,
                    "score": score,
                    "box": box.tolist() # [x1, y1, x2, y2]
                })
        
        return results

    def add_dataset_image(self, filename):
        annotations = json.load(open(self.annot_path))
        
        for image in annotations["images"]:
            if image["file_name"] == filename:
                print("Image already exists in dataset")
                return
        
        im = cv2.imread(filename)
        window_name = "Select area"
        cv2.namedWindow(window_name)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
        area = cv2.selectROI(window_name, im, fromCenter=False, showCrosshair=False)
        cv2.destroyWindow(window_name)
        try:
            category = int(input("Enter category: "))
        except ValueError:
            print("Invalid category.")
            return

        annotations["images"].append({
            "id": len(annotations["images"]) + 1,
            "width": im.shape[1],
            "height": im.shape[0],
            "file_name": filename,
            "date_captured": "2025-12-14T15:32:00+00:00"
        })
        
        annotations["annotations"].append({
            "id": len(annotations["annotations"]) + 1,
            "image_id": len(annotations["images"]),
            "category_id": category,
            "segmentation": [],
            "area": area[2] * area[3],
            "bbox": [area[0], area[1], area[2], area[3]],
            "iscrowd": 0
        })

        json.dump(annotations, open(self.annot_path, "w"))
        print("Success!")

    def clear_dataset(self):
        annotations = json.load(open(self.annot_path))
        annotations["images"] = []
        annotations["annotations"] = []
        json.dump(annotations, open(self.annot_path, "w"))

    def select_folder(self):
        tk.Tk().withdraw()
        root = tk.Tk()
        root.attributes("-topmost", True)
        root.withdraw()
        folder_path = askdirectory(parent=root, initialdir=self.BASE_DIR)
        root.destroy()
        return folder_path

    def add_folder_images(self, folder_path=None):
        if not folder_path:
            folder_path = self.select_folder()
        if not folder_path:
            return

        dest_dir = self.train_dir
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)

        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                start_name, ext = os.path.splitext(filename)
                if start_name.endswith("_added"):
                    print(f"Skipping {filename} (already added)")
                    continue

                src_path = os.path.join(folder_path, filename)
                dest_path = os.path.join(dest_dir, filename)

                if not os.path.exists(dest_path):
                    with Image.open(src_path) as img:
                        max_dim = 640
                        if img.width > max_dim or img.height > max_dim:
                            ratio = min(max_dim / img.width, max_dim / img.height)
                            new_size = (int(img.width * ratio), int(img.height * ratio))
                            img = img.resize(new_size, Image.Resampling.LANCZOS)
                            print(f"Resized {filename} to {new_size}")
                        
                        img.save(dest_path)
                        print(f"Saved {filename} to training folder.")
                else:
                    print(f"{filename} already in training folder.")

                print(f"Processing {filename}...")
                self.add_dataset_image(dest_path)
                
                new_src_filename = f"{start_name}_added{ext}"
                new_src_path = os.path.join(folder_path, new_src_filename)
                try:
                    os.rename(src_path, new_src_path)
                    print(f"Renamed original to {new_src_filename}")
                except OSError as e:
                    print(f"Error renaming {filename}: {e}")

    def select_image(self):
        tk.Tk().withdraw()
        root = tk.Tk()
        root.attributes("-topmost", True) 
        root.withdraw()
        filename = askopenfilename(parent=root, initialdir=self.train_dir)
        root.destroy()
        return filename

def clear_screen():
    os.system("cls")

def main():
    detector = CrabDetector()
    
    while True:
        clear_screen()
        mode = input(""" 
ds - dataset mode
tn - training mode
t - test mode
e - exit
Enter mode: """)
        
        if mode == "ds":
            # Dataset Mode Submenu
            clear_screen()
            command = input("""
s - select image
sf - select folder
c - clear dataset
e - exit
Enter command: """)
            if command == "s":
                filename = detector.select_image()
                if filename:
                    detector.add_dataset_image(filename)
            elif command == "sf":
                detector.add_folder_images()
            elif command == "c":
                if input("Are you sure you want to clear the dataset? (y/n) ") == "y":
                    detector.clear_dataset()
                else:
                    print("Cancelled")
            elif command == "e":
                exit()
            else:
                print("Invalid command")

        elif mode == "tn":
            # Train Mode
            detector.train(epochs=10)

        elif mode == "t":
            # Test Mode
            img_path = detector.select_image()
            if not img_path:
                continue
                
            results = detector.detect(img_path)
            
            # Visualization Logic (kept here as requested to separate logic from class core return)
            image_bgr = cv2.imread(img_path)
            crab_num = 0
            
            for res in results:
                label = res['label']
                score = res['score']
                box = res['box'] # [x1, y1, x2, y2]
                
                crab_num += 1
                
                # Draw
                cv2.putText(image_bgr, f"{label}: {score:.2f}", (box[0], box[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.rectangle(image_bgr, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
                
                print(f"Detected: {label} at position: {box}")

            # Show image
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(16, 12))
            plt.imshow(image_rgb)
            plt.axis('off')
            plt.show() # Blocks

            print("Number of crabs detected:", crab_num)
            if crab_num == 0:
                print("No crabs detected")
            
            time.sleep(10)

        elif mode == "e":
            exit()
        else:
            print("Invalid mode")

if __name__ == "__main__":
    main()
