from ultralytics import YOLO


if __name__ == "__main__":
    # 加载YOLOv8模型
    model = YOLO("/home/featurize/YOLOv8-loopy-main/best.pt")
    # 视频路径
    file_path = "/home/featurize/YOLOv8-loopy-main/当男朋友第一次去小loopy家.mp4"
    # 检测视频
    results = model.predict(source=file_path, device=0, show=False, save=True)
