import yaml
from ultralytics import YOLO


# 1. 데이터 설정을 YAML 파일로 저장
data_config = {
    'train': './dataset/images/train',  # 학습 이미지 경로
    'val': './dataset/images/val',      # 검증 이미지 경로
    'nc': 3,                            # 클래스 개수
    'names': ['circle_point', 'cable', 'knitting_part']  # 클래스 이름
}

with open('dataset.yaml', 'w') as f:
    yaml.dump(data_config, f)

# 2. 모델 초기화 (YOLOv8n: Nano 모델)
model = YOLO('yolov8m.yaml')  # Nano 모델 기반

# 가중치 조정 (circle_point에 높은 가중치 부여)
class_weights = {
    'circle_point': 2.0,  # 2배 더 높은 가중치
    'cable': 1.0,
    'knitting_part': 1.0
}

# 3. 모델 학습
model.train(
    data='dataset.yaml',       # YAML 파일 경로 전달
    epochs=100,                 # 학습 반복 횟수
    batch=16,                  # 배치 크기
    imgsz=640,                 # 입력 이미지 크기
    workers=4,                 # 멀티프로세싱 개수
    project='knitting_project_yoloMideum',  # 프로젝트 디렉토리
    name='cable_detection',      # 모델 이름
    device='0',                  # GPU 사용 (0번 GPU)
    resume=True,
    augment=True,
    lr0=0.01,
    optimizer='AdamW',
    # # auto_anchor=True,
    # auto_augment=randaugment,
    cls=class_weights['circle_point'],  # circle_point의 손실 비중 강화
    kobj=1.0,  # 객체 가중치
    # iou=0.6
)

# # 4. 평가 및 결과 확인
# results = model.val()
# print(results)

# # 5. 학습된 모델 로드 및 테스트
# model = YOLO('./knitting_project_yoloSmall/cable_detection/weights/best.pt')
# results = model('./test_images/sample.jpg')
# results[0].save(save_dir='./output')  # 결과 저장
