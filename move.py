import shutil
import os

file_List = ["train", "valid", "test"]
DATASET_PATH = "../Datasets/facemask/"   # facemask数据集的路径

for file in file_List:
    # 创建VOC文件夹
    if not os.path.exists('../VOC/images/%s' % file):
        os.makedirs('../VOC/images/%s' % file)
    if not os.path.exists('../VOC/labels/%s' % file):
        os.makedirs('../VOC/labels/%s' % file)
    f = open(DATASET_PATH + '%s.txt' % file, 'r')
    lines = f.readlines()
    # 按照YOLOv5格式复制数据集文件到VOC中
    for line in lines:
        line = DATASET_PATH+"/".join(line.split('/')[-3:]).strip()
        shutil.copy(line, "../VOC/images/%s" % file)
        line = line.replace('JPEGImages', 'labels')
        origin = line.split('.')[2:-1]
        extension = line.split('.')[-1]
        label_path = ".."+".".join([".".join(origin).strip(),"txt"])
        print(label_path)
        shutil.copy(label_path, "../VOC/labels/%s/" % file)
