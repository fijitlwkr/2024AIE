import yaml
from ultralytics import YOLO


# 데이터 설정을 YAML 파일로 저장
data_config = {
    'train': './processed_data/train',  # 학습 이미지 경로
    'val': './processed_data/test',      # 검증 이미지 경로
    'nc': 3,                            # 클래스 개수
    'names': ['cable', 'knitting_part', 'circle_point']  # 클래스 이름
}

with open('dataset.yaml', 'w') as f:
    yaml.dump(data_config, f)
    
########## 모델 학습 ##########

# 1. YOLO 모델 초기화 (Pre-trained 모델 활용)
model = YOLO('yolov8s.pt')  # YOLOv8 Small 버전

# 2. 가중치 조정
class_weights = {
    'circle_point': 2.0,  # circle_point에 높은 가중치
    'cable': 1.0,
    'knitting_part': 1.0
}

# 3. 모델 학습
model.train(
    data='./dataset.yaml',       # YAML 파일 경로 전달
    epochs=500,                 # 학습 반복 횟수
    batch=16,                    # 배치 크기
    imgsz=640,                   # 입력 이미지 크기
    workers=4,                   # 멀티프로세싱 개수
    project='./attribute_yoloSmall',  # 프로젝트 디렉토리
    name='cable_detection',      # 모델 이름
    device='0',                  # GPU 사용 (0번 GPU)
    resume=False,                # 학습 재개 여부
    augment=True,                # 데이터 증강 활성화
    lr0=0.01,                    # 초기 학습률
    optimizer='AdamW',           # 옵티마이저 설정
    cls=class_weights['circle_point'],  # circle_point 클래스에 높은 가중치
    kobj=1.0,                     # 객체 손실 가중치
    patience=0
)