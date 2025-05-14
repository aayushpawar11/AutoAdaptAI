import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision import transforms
from PIL import Image, ImageDraw
import os

#Pretrained model
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter',
    'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
    'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

def get_transform():
    return transforms.Compose([transforms.ToTensor()])

def detect_objects_in_folder(input_folder, output_folder, confidence_threshold=0.3):
    transform = get_transform()

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for img_file in os.listdir(input_folder):
        img_path = os.path.join(input_folder, img_file)
        image = Image.open(img_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            predictions = model(image_tensor)[0]

        draw = ImageDraw.Draw(image)
        for idx, score in enumerate(predictions['scores']):
            if score > confidence_threshold:
                label_idx = predictions['labels'][idx].item()
                if label_idx < len(COCO_INSTANCE_CATEGORY_NAMES):
                    box = predictions['boxes'][idx].tolist()
                    label = COCO_INSTANCE_CATEGORY_NAMES[label_idx]
                    draw.rectangle(box, outline='red', width=2)
                    draw.text((box[0], box[1]), label, fill='red')
                    print(f"Detected: {label} (score: {score:.2f})")
                else:
                    print(f"⚠️ Skipping unknown label index: {label_idx}")

        image.save(os.path.join(output_folder, f"annotated_{img_file}"))
        print(f"Detected: {label} (score: {score:.2f})")

if __name__ == "__main__":
    detect_objects_in_folder("data/frames", "data/annotated", confidence_threshold=0.3)
