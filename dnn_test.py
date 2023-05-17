import cv2

from models.experimental import *
from utils.datasets import *
from utils.general import *
import time

WEIGHTS_PATH = "./runs/train/ghost/weights/best.pt"
VIDEOS_PATH = "../DataSets/videos/facemask_detection.mp4"
IMAGE_PATH = "../DataSets/face.jpg"

def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def transImg(img0, img_size=640):
    img = letterbox(img0, new_shape=img_size)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
    img = np.ascontiguousarray(img)
    return img

# 初始化设备
device = torch.device('cuda:0')

# 加载pt模型
model = attempt_load(WEIGHTS_PATH, map_location=device)  # load FP32 model
model.half() # uint8类型转为float16/32半精度浮点类型

# 从模型中获取模型的分类名称
names = model.module.names if hasattr(model, 'module') else model.names
# 随机为不同的分类名称生成不同颜色
colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

capture = cv2.VideoCapture()
capture.open(VIDEOS_PATH)   # 加载视频

while True:
    ret, frame = capture.read()
    start = time.time()
    if not ret:
        break
    img = transImg(frame)    # 改变图像尺寸大小、并BGR转RGB
    img = torch.from_numpy(img).to(device)  # numpy数组转换成tensor张量
    img = img.half()  # uint8类型转为float16/32半精度浮点类型
    img /= 255.0  # 归一化处理

    if img.ndimension() == 3:   # 判断图像维度是否为3
        img = img.unsqueeze(0)  # 在第1个维度上增加一个维度

    # 图像预测
    pred = model(img, augment=False)[0]

    # 非极大值抑制处理
    pred = non_max_suppression(pred, 0.4, 0.5)



    # 预测结果可视化绘制
    for i, det in enumerate(pred):  # 遍历检测到的结果
        if det is not None and len(det):
            # print("Face mask detection count: %i" % len(det))

            text = 'Face mask detection count: %i' % len(det)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            thickness = 4
            (x, y), _ = cv2.getTextSize(text, font, font_scale, thickness)

            # 将文字放在右上角，留出一定的边距
            text_offset_x = frame.shape[1] - x - 10
            text_offset_y = y + 10

            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
            # 绘制预测结果
            for *xyxy, conf, cls in det:
                label = '%s %.2f' % (names[int(cls)], conf)
                plot_one_box(xyxy, frame, label=label, color=colors[int(cls)], line_thickness=1)

            cv2.putText(frame, text, (text_offset_x, text_offset_y), font,
                        font_scale, (0,0,255), thickness)

        # print("花费时间: %f 秒" % round(time.time() - start, 4))
        cv2.imshow("Face Mask Detection", frame)
        k = cv2.waitKey(1) & 0xFF
        if (k == 113):  # 按键 q 退出
            break

capture.release()
cv2.destroyAllWindows()