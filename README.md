### 基于YOLO-V8的Loopy（某动漫角色）的识别
![1713533874344](image/README/1713533874344.png)
#### 1、准备工作
先在网络上搜寻很多loopy的图片，然后将图片导入Lablel Studio软件进行标注，并导出yolo格式。
本人大约导入了300张图片，并完成了手动标注工作。
#### 2、训练模型
本人租用的服务器为Tesla v100 16G，选用的yolov8版本为yolov8n
#### 3、经验
首先只有300张效果其实就很好了，然后我训练了1000轮，batchsize开了64，大约吃10G左右显存，可以调整（我训练了大约半个小时），最后效果还是很好打，但是根据评估，600轮就够了
![loopy](image/README/loopy.png)
![F1_curve](image/README/F1_curve.png)
