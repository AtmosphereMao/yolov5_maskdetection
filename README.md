# YOLOv5 Mask Detection

对
[YOLOv5](https://github.com/ultralytics/yolov5)
进行MobileNets改进及加入注意力机制等优化，引入OpenCV对口罩佩戴进行检测的系统。

~~*此为毕设项目、暂未完成改进*~~

## 数据集

[Face Mask Dataset (YOLO Format)](https://www.kaggle.com/datasets/aditya276/face-mask-dataset-yolo-format)

## 运行方式

- 安装依赖
    - `pip install -r requirements.txt`

- 初始化数据集
    - 修改`move.py`中数据集目录路径`DATASET_PATH`
    - `python move.py` 处理数据集
    - 修改`data/mask.yaml`数据集目录路径为处理后数据集的路径

- 训练结果
    - 修改`train.py`的参数（可选，已调好参数）
    - `python train.py`

- 运行检测
    - `python dnn_test.py`