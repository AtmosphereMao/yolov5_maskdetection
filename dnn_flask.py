import cv2

import db_tool
from models.experimental import *
from utils.datasets import *
from utils.general import *
from flask import Flask, render_template, Response

app = Flask(__name__)

WEIGHTS_PATH = "./runs/train/ghost/weights/best.pt"
VIDEOS_PATH = "../DataSets/videos/facemask_detection.mp4"
RESULT_SAVE_PATH = "../results/mask_detection/"


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        if label == "no_mask":
            label = "未佩戴口罩"
        elif label == "mask":
            label = "已佩戴口罩"
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


def gen_display(monitor_id):
    # 初始化设备
    device = torch.device('cuda:0')

    # 加载pt模型
    model = attempt_load(WEIGHTS_PATH, map_location=device)  # load FP32 model
    model.half()  # uint8类型转为float16/32半精度浮点类型

    # 从模型中获取模型的分类名称
    names = model.module.names if hasattr(model, 'module') else model.names
    # 随机为不同的分类名称生成不同颜色
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # 初始化Tracker
    tracker = cv2.TrackerTLD_create()

    # 初始化DB
    db = db_tool.DB()

    if monitor_id == 1:
        capture = cv2.VideoCapture()
        capture.open(VIDEOS_PATH)  # 加载视频
    else:
        capture = cv2.VideoCapture(0)

    i = 0

    while True:
        ret, frame = capture.read()

        # 重置帧数
        i += 1
        if i == int(capture.get(cv2.CAP_PROP_FRAME_COUNT)):
            i = 0
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

        if not ret:
            break
        img = transImg(frame)  # 改变图像尺寸大小、并BGR转RGB
        img = torch.from_numpy(img).to(device)  # numpy数组转换成tensor张量
        img = img.half()  # uint8类型转为float16/32半精度浮点类型
        img /= 255.0  # 归一化处理

        if img.ndimension() == 3:  # 判断图像维度是否为3
            img = img.unsqueeze(0)  # 在第1个维度上增加一个维度

        # 图像预测
        pred = model(img, augment=False)[0]

        # 非极大值抑制处理
        pred = non_max_suppression(pred, 0.4, 0.5)

        # 预测结果可视化绘制
        for i, det in enumerate(pred):  # 遍历检测到的结果
            if det is not None and len(det):
                # print("Face mask detection count: %i" % len(det))
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
                # 绘制预测结果
                for *xyxy, conf, cls in det:
                    temp = (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]))
                    # tracker.init(frame, temp)
                    # ok, bbox = tracker.update(frame)
                    #
                    # if not ok:  # 未检测过
                    # if i % 5 == 0:
                    #     signTimestamp = str(int(round(time.time() * 1000)))
                    #     h = hashlib.md5()
                    #     h.update(signTimestamp.encode(encoding='utf-8'))
                    #     path = RESULT_SAVE_PATH + h.hexdigest() + ".jpg"
                    #     cv2.imwrite(path, frame[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])])

                    label = '%s %.2f' % (names[int(cls)], conf)
                    plot_one_box(xyxy, frame, label=label, color=colors[int(cls)], line_thickness=3)
                """
                视频流生成器功能。
                """
                # 读取图片
                # frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                # 将图片进行解码
                ret, frame = cv2.imencode('.jpeg', frame)
                if ret:
                    # 转换为byte类型的，存储在迭代器中
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame.tobytes() + b'\r\n')


@app.route('/detection/<int:nid>')
def detection(nid):
    return Response(gen_display(nid),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
    #         cv2.imshow("Face Mask Detection", frame)
    #         k = cv2.waitKey(1) & 0xFF
    #         if (k == 113):  # 按键 q 退出
    #             break
    #
    # capture.release()
    # cv2.destroyAllWindows()


@app.route('/')
def index():
    # return the rendered template
    return render_template("index.html")


if __name__ == '__main__':
    app.run(host="127.0.0.1", port="8000", debug=True,
            threaded=True, use_reloader=False)
