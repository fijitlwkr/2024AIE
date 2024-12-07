import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms
from timm import create_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm  # Progress bar for training
import cv2

# 경로 설정
cropped_images_dir = "./processed_data/cropped_images"  # 크롭된 이미지 디렉토리
output_images_dir = "./EffNet_attribute/output_images_b3"  # 시각화 결과 저장 경로
model_save_path = "./efficientnet_weights/best.pth"  # 모델 가중치 저장 경로
os.makedirs(output_images_dir, exist_ok=True)

# EfficientNet-Lite 모델 정의
class CableAttributeModel(nn.Module):
    def __init__(self, pretrained=False):
        super(CableAttributeModel, self).__init__()
        self.backbone = create_model('efficientnet_b3', pretrained=pretrained)
        in_features = self.backbone.num_features
        self.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2)  # [num_of_twist, num_of_cols]
        )
    
    def forward(self, x):
        x = self.backbone.forward_features(x)
        x = nn.AdaptiveAvgPool2d((1, 1))(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# 데이터셋 클래스 정의
class CableDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform if transform else transforms.ToTensor()
        self.samples = self._load_data()

    def _load_data(self):
        samples = []
        for folder_name in os.listdir(self.data_dir):
            folder_path = os.path.join(self.data_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue

            attributes_path = os.path.join(folder_path, "attributes.txt")
            with open(attributes_path, "r") as f:
                attributes = f.readlines()
                num_of_twist = int(attributes[0].split(":")[1].strip())
                num_of_cols = int(attributes[1].split(":")[1].strip())

            for image_name in os.listdir(folder_path):
                if image_name.endswith(".jpg") and "twist" in image_name:
                    image_path = os.path.join(folder_path, image_name)
                    samples.append((image_path, num_of_twist, num_of_cols))
        return samples

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image_path, num_of_twist, num_of_cols = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        labels = torch.tensor([num_of_twist, num_of_cols], dtype=torch.float32)
        return image, labels

# 데이터 준비
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dataset = CableDataset(cropped_images_dir, transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# 모델, 손실 함수, 옵티마이저 정의
model = CableAttributeModel(pretrained=True)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 학습 및 검증 루프
best_val_loss = float('inf')
for epoch in range(50):  # 에폭 수 조정 가능
    model.train()
    train_loss = 0
    for images, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0
    ground_truth_twist = []
    predicted_twist = []
    ground_truth_cols = []
    predicted_cols = []
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            # 성능 평가 데이터 수집
            ground_truth_twist.extend(labels[:, 0].tolist())
            predicted_twist.extend(outputs[:, 0].tolist())
            ground_truth_cols.extend(labels[:, 1].tolist())
            predicted_cols.extend(outputs[:, 1].tolist())
    val_loss /= len(val_loader)

    print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # 모델 저장
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), model_save_path)
        print(f"Best model saved with validation loss: {best_val_loss:.4f}")

# 성능 평가
twist_mse = mean_squared_error(ground_truth_twist, predicted_twist)
twist_mae = mean_absolute_error(ground_truth_twist, predicted_twist)
cols_mse = mean_squared_error(ground_truth_cols, predicted_cols)
cols_mae = mean_absolute_error(ground_truth_cols, predicted_cols)

print(f"num_of_twist - MSE: {twist_mse:.4f}, MAE: {twist_mae:.4f}")
print(f"num_of_cols - MSE: {cols_mse:.4f}, MAE: {cols_mae:.4f}")

# 시각화 및 결과 저장
for idx, (image_path, num_of_twist_gt, num_of_cols_gt) in enumerate(dataset.samples):
    if idx >= 10:  # 시각화할 샘플 수 제한 (예: 10개)
        break
    image = cv2.imread(image_path)
    num_of_twist_pred = predicted_twist[idx]
    num_of_cols_pred = predicted_cols[idx]

    # Ground Truth 및 예측 값 표시
    cv2.putText(image, f"Twist GT: {num_of_twist_gt}, Pred: {int(round(num_of_twist_pred))}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(image, f"Cols GT: {num_of_cols_gt}, Pred: {int(round(num_of_cols_pred))}", 
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    output_image_path = os.path.join(output_images_dir, f"sample_{idx}.jpg")
    cv2.imwrite(output_image_path, image)

print(f"시각화된 결과가 {output_images_dir}에 저장되었습니다.")
