import os
import numpy as np
import torch
import torchvision
from torchvision import transforms
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from torchvision.transforms import functional as F

# ---------------------- 0. 自定义数据增强类 ----------------------
class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class ToTensor:
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target

class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, target):
        if torch.rand(1) < self.p:
            image = F.hflip(image)
            if "boxes" in target:
                boxes = target["boxes"]
                boxes[:, [0, 2]] = image.size(2) - boxes[:, [2, 0]]
                target["boxes"] = boxes
            if "masks" in target:
                target["masks"] = torch.flip(target["masks"], [-1])
        return image, target

def get_transform(train):
    transforms_list = []
    transforms_list.append(ToTensor())
    if train:
        transforms_list.append(RandomHorizontalFlip(0.5))
    return Compose(transforms_list)

# ---------------------- 1. 数据集定义 ----------------------
class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        mask = np.array(mask)

        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]

        masks = mask == obj_ids[:, None, None]
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])


        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

# ---------------------- 2. 模型定义 ----------------------
def get_model_instance_segmentation(num_classes):
    model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.COCO_V1)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model

# ---------------------- 3. collate_fn ----------------------
def collate_fn(batch):
    return tuple(zip(*batch))

# ---------------------- 4. 训练函数 ----------------------
def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10):
    model.train()
    for i, (images, targets) in enumerate(data_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if i % print_freq == 0:
            print(f"Epoch {epoch}, Iteration {i}, Loss: {losses.item():.4f}")

# ---------------------- 5. 可视化函数（已修复） ----------------------
def visualize(image, target, prediction=None):
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    ax = plt.gca()

    if target is not None:
        boxes = target["boxes"].numpy()
        masks = target["masks"].numpy()
        for box, mask in zip(boxes, masks):
            rect = Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],
                             fill=False, color='green', linewidth=2)
            ax.add_patch(rect)
            plt.imshow(mask, alpha=0.3, cmap='viridis')

    if prediction is not None:
        boxes = prediction["boxes"].cpu().numpy()
        scores = prediction["scores"].cpu().numpy()
        masks = prediction["masks"].cpu().numpy()

        keep = scores > 0.5
        boxes = boxes[keep]
        masks = masks[keep]

        for box, mask in zip(boxes, masks):
            rect = Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],
                             fill=False, color='red', linewidth=2)
            ax.add_patch(rect)
            plt.imshow(mask[0], alpha=0.3, cmap='plasma')

    plt.axis('off')
    plt.show()
    # ---------------------- 7. 主程序 (修正版) ----------------------
if __name__ == "__main__":
    data_root = r"C:\Users\asus\Desktop\myproject\PennFudanPed\PennFudanPed"
    num_classes = 2
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("使用设备:", device)

    # 数据集
    dataset = PennFudanDataset(data_root, transforms=get_transform(train=True))
    dataset_test = PennFudanDataset(data_root, transforms=get_transform(train=False))

    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=0, collate_fn=collate_fn
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn
    )

    model = get_model_instance_segmentation(num_classes).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    num_epochs = 10
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        lr_scheduler.step()

    # ========== 这里开始：显示 纯原图 + 标注 + 预测 ==========
    model.eval()
    with torch.no_grad():
        for i, (images, targets) in enumerate(data_loader_test):
            # 模型输入
            images = [img.to(device) for img in images]
            outputs = model(images)

            # ======================
            # 关键：拿 原始 PIL 原图
            # ======================
            # 从 dataset_test 里拿真实原图
            idx = dataset_test.indices[i]
            full_dataset = dataset_test.dataset
            img_path = os.path.join(full_dataset.root, "PNGImages", full_dataset.imgs[idx])
            original_img = Image.open(img_path).convert("RGB")

            # 显示纯原图
            plt.figure()
            plt.imshow(original_img)
            plt.title("原始纯原图")
            plt.axis("off")
            plt.show()

            # 再显示 原图+bbox+mask
            visualize(original_img, targets[0], outputs[0])

            if i > 5:
                break