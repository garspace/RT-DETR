import cv2,glob
from math import *
import numpy as np
import os, random, shutil
import glob as gb
from time import sleep
import copy
import json

class JsonEncoder(json.JSONEncoder):
    """Convert numpy classes to JSON serializable objects."""
 
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(JsonEncoder, self).default(obj)

def copyFile2Folder(srcfile,dstfolder):
    '''
    复制文件到指定文件夹，名字和以前相同
    Args:
        srcfile: '04_spur_06.txt'  文件的绝对路径
        dstfile: 'labels'  文件夹
    Returns:
    '''
    if not os.path.isfile(srcfile):
        print("%s not exist!" % (srcfile))
    else:
        src_fpath, src_fname = os.path.split(srcfile)  # 分离文件名和路径
        if not os.path.exists(dstfolder):
            os.makedirs(dstfolder)  # 创建路径dst_file
        dst_file =os.path.join(dstfolder, src_fname)
        shutil.copyfile(srcfile, dst_file)  # 复制文件
        print ("copy %s -> %s" % (srcfile, dst_file))
        return dst_file

class cocoJson(object):
    '''
    coco 的json 的文件格式类
    '''
    def __init__(self,categories):
        self.info = {'description': 'jingduan DATASET',
                'url': 'DLH',
                'version': '1.0',
                'year': 2023,
                'contributor': 'dengjie',
                'date_created': '2023-09-12 16:11:52'
                }
        self.license = {
        "url": "none",
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License"}
        self.images = None
        self.annotations = None
        self.category = categories
        self.cocoJasonDict = {"info": self.info, "images": self.images, "annotations" : self.annotations, "licenses" : self.license,'categories':self.category}
	
    def getDict(self):
        '''
        Returns: 返回 格式的字典化
        '''
        self.cocoJasonDict = {"info": self.info, "images": self.images, "annotations": self.annotations,
                              "licenses": self.license,'categories':self.category}
        return self.cocoJasonDict


if __name__ == '__main__':
    # 文件原本：
    '''
    root: DATASET
                    ------------------->labels/  # 原本的所有目标检测框的,格式： *.txt 或者 *.json
                    ------------------->images/   #  *.jpg  所有的图片
    最终：
    root: ../DATASET_COCO   
                        ------------------->images/   #  *.jpg  所有的图片
                        ------------------->annotations  /  instances_train_set_name.json   # 存放 labels 下所有对应的train.json
                                                         /  instances_val_set_name.json     # 存放 labels val.json
    '''

    data_root = './data2/'
    dataset_name = 'jingduan_good_coco'
    is_oneobj = True
    if is_oneobj:
        dataset_name = dataset_name + '_oneobj'

    # 保持train 或者 Val 
    #mode = 'val'

    # 获取数据集所有类别
    className = []
    for path in glob.glob(data_root+'**/*.jpg'):
        json_path = path.replace('.jpg','.json')
        with open(json_path,'r') as ff:
            obj = json.load(ff)
            for k,v in obj.items():
                if 'shape' in k:
                    for ll in obj[k]:
                        points = ll['points']
                        box_type = ll['shape_type']
                        label = ll['label']
                        if label not in className:
                            className.append(label)
    if is_oneobj:
        className = ['object']
    print('classes:', className)

    # 存放 train.txt  和  val.txt
    for mode in ['train','val']:
        Imageset =  dataset_name + '_' + mode + '.txt'
        with open(Imageset,'w',encoding='utf-8') as f:
            for path in glob.glob(data_root+ mode +'/*.jpg'):
               f.write(path+'\n')
        # 保持原图到target目录
        target =  os.path.join(dataset_name, 'images')
        
        # 生成annotation json 文件  
        jsonFile = os.path.join(dataset_name,'annotation', 'instances_' + mode + '.json')
        if not os.path.exists(os.path.join(dataset_name,'annotation')):
            os.makedirs(os.path.join(dataset_name,'annotation'))
        print(f'jsonFile saved {jsonFile}')

        # 检查目标文件夹是否存在
        if not os.path.exists(target):
            os.makedirs(target)

        # images 段 的字典模板
        images = {	"license":3,
            "file_name":"COCO_val2014_000000391895.jpg",
            "coco_url":"",
            "height":360,"width":640,"date_captured":"2013-11-14 11:18:45",
            "id":0 }

        # annotation 的字典模板
        an = {"segmentation": [],
            "iscrowd": 0,
            "keypoints": [],
            "area": 0,
            "image_id": 0, "bbox": [], "category_id": 0,
            "id": 0}

        # categories 的字典模板
        cate_ = {
            'id':0,
            'name':'a',
        }

        # 用来保存目标类的字典
        cate_list = []
        temId = 0
        for idName in className:
            tempCate = cate_.copy()
            tempCate['id'] = temId
            temId += 1
            tempCate['name'] = idName
            cate_list.append(tempCate)

        js = cocoJson(cate_list)
        image_lsit = []
        annoation_list =[]

        with open(Imageset, 'r') as f:
            lines = f.readlines()

        img_id = 0
        bbox_id = 0
        # 按值打开图片
        for path in lines:
            path = path.lstrip().rstrip()
            image = cv2.imread(path)
            # 将图片副知道新的文件夹
            copyFile2Folder(path,target)
            # 得到宽高
            (height, width) = image.shape[:2]
            # 得到文件名子
            _,fname = os.path.split(path)

            # 图像对应的txt 文件路径
            txtPath = path.replace('.jpg','.json')
            if not os.path.exists(txtPath):
                txtPath = path.replace('.jpg','.txt')
            #txtPath = txtPath.replace('image', 'labels')
            # 复制images 的字典的复制        
            image_temp = images.copy()
            image_temp['file_name'] = fname
            image_temp['height'] = height
            image_temp['width'] = width
            image_temp['id'] = img_id
            image_lsit.append(image_temp)
            # 打开图片的对应的txt标注文件
            if not os.path.exists(txtPath):
                with open(txtPath,'r') as re:
                    txtlines = re.readlines()
                    for txtline in txtlines:
                        temp = txtline.rstrip().lstrip().split(' ')
                        # 目标的类 中心值 xy  和  该检测框的宽高
                        classid = int(temp[0])
                        x = float(temp[1]) * width
                        y = float(temp[2]) * height
                        w = float(temp[3]) * width
                        h = float(temp[4]) * height
                        # 复制annotation 的字典 
                        temp_an = an.copy()
                        temp_an['image_id'] = img_id
                        temp_an['bbox'] = [x,y,w,h]
                        temp_an['category_id'] = classid
                        temp_an['id'] = bbox_id
                        bbox_id += 1 # 这个是 这个annotations 的id 因为一个图像可能对应多个 目标的id
                        annoation_list.append(temp_an)
            else:
                # 打开图片的对应的json 标注文件
                with open(txtPath,'r') as fr:
                    obj = json.load(fr)
                    for k,v in obj.items():
                        if 'shape' in k:
                            for ll in obj[k]:
                                points = ll['points']
                                box = np.array(points).reshape(-1,2).astype(np.int32)
                                #print(points)
                                #rbox = cv2.minAreaRect(np.array(points).reshape(-1,2).astype(np.int32)) 
                                #box = np.int0(cv2.boxPoints(rbox))
                                xmin = min(box[:,0])
                                xmax = max(box[:,0])
                                ymin = min(box[:,1])
                                ymax = max(box[:,1])
                                x = xmin #(xmin + xmax)/2
                                y = ymin #(ymin + ymax)/2
                                w = xmax - xmin
                                h = ymax - ymin
                                box_type = ll['shape_type']
                                label = ll['label']
                                if is_oneobj:
                                    label = 'object'
                                    classid = className.index(label)
                                else:
                                    classid = className.index(label)

                                temp_an = an.copy()
                                temp_an['area'] = height*width
                                temp_an['image_id'] = img_id
                                temp_an['bbox'] = [x,y,w,h]
                                temp_an['category_id'] = classid
                                temp_an['id'] = bbox_id
                                bbox_id += 1 # 这个是 这个annotations 的id 因为一个图像可能对应多个 目标的id
                                annoation_list.append(temp_an)
            # 图像的id
            img_id += 1

        # 将json 的实例 annotations  赋值
        js.images = image_lsit
        js.annotations = annoation_list
        # 写入文件
        json_str = json.dumps(js.getDict(),cls=JsonEncoder)
        with open(jsonFile,'w+') as ww:
            ww.write(json_str)
        with open(jsonFile.replace('.json','.txt'),'w') as txt:
            catID=js.getDict()['categories']
            for ob in catID:
                id_val=''
                for k,v in ob.items():
                    if k=='id':
                        txt.write(k+':')
                    elif v=='name':
                        txt.write(v+'\n')
        print('finished')
