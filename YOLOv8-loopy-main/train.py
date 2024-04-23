from ultralytics import YOLO

if __name__ == "__main__":
    # 加载预训练模型
    model = YOLO("yolov8n.pt")

    # 开始训练
    model.train(
        data="/home/featurize/YOLOv8-loopy-main/dataset/loopy.yaml",
        epochs=1000,
        batch=64,
        imgsz=640,
        device="0",
    )
