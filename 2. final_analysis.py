#!/usr/bin/env python
# coding: utf-8

# In[1]:


from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
import os
import pandas as pd
import datetime as dt
from matplotlib.pyplot import imread
from keras.models import load_model
from PIL import Image
from tqdm import tqdm_notebook
import scipy
from keras.models import load_model
import itertools
import cx_Oracle


# # 함수 정의

# ### load_images(경로, 파일이름) : 이미지 파일 불러오기
# - pred()에 사용

# In[2]:


def load_images(default_path, file_names):
    images = []

    for file_name in file_names:
        filepath = default_path + file_name
        
        # 파일 읽기
        image = imread(filepath)
        # 이미지 파일 리스트에 추가
        images.append(image) 

    # 이미지 파일 numpy 배열로 변환
    images = np.array(images) 
    return images


# ### load_images_cv(경로, 파일이름) : 이미지 파일 불러오기
# - pepper_att(), size(), color_pct()에 사용

# In[16]:


def load_images_cv(default_path, file_names):
    images = []

    for file_name in file_names:
        filepath = default_path + file_name

        image = cv2.imread(filepath)
        images.append(image)

    images = np.array(images)

    return images


# ### resize_image(이미지, (가로사이즈, 세로사이즈)) : 리사이즈
# - 이미지는 배열형태이어야 함!

# In[3]:


def resize_image(original_images, size):
    resized_images = []
    
    for image in original_images:
        resized_image = np.array(Image.fromarray(image).resize(size))
        resized_images.append(resized_image)
        
    resized_images = np.array(resized_images)
    
    return resized_images


# ### size(images) : 가로 세로 길이구하기

# In[4]:


def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


# In[5]:


def size(images):
    
    result = []
    
    for image in images:
    
        size_list = []
        
        # 기준선(5cm) 추가
        cv2.line(image,(10,10),(100,10),(0,0,0),1)
        # 흑백으로 바꾸고 약간의 블러처리하기
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        # 가장자리 찾고, dilation + erosion 하기
        edged = cv2.Canny(gray, 50, 100)
        edged = cv2.dilate(edged, None, iterations=1)
        edged = cv2.erode(edged, None, iterations=1)

        # 윤곽이 있는 부분 찾기
        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        # 왼쪽 이미지 -> 오른쪽 이미지 순으로 정렬하기
        (cnts, _) = contours.sort_contours(cnts)
        pixelsPerMetric = None 

        # 첫번째 윤곽선(5cm짜리 선) 제외하고 윤곽선 면적순으로 정렬하기
        n_cnts = list(cnts)
        n_cnts[1:] = sorted(n_cnts[1:], key=lambda x: cv2.contourArea(x), reverse=True)

        cnts = tuple(n_cnts)

        # 첫번째 윤곽선(5cm짜리 선)과 두번째 윤곽선(고추) 길이 구하기
        for c in cnts[:2]:
            # 윤곽선의 박스 구하기
            orig = image.copy()
            box = cv2.minAreaRect(c)
            box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
            box = np.array(box, dtype="int")

            # 박스의 꼭지점 정렬하기(top-left, top-right, bottom-right, and bottom-left)
            box = perspective.order_points(box)

            # 두 꼭지점의 중간점 구하기
            (tl, tr, br, bl) = box
            (tltrX, tltrY) = midpoint(tl, tr)
            (blbrX, blbrY) = midpoint(bl, br)
            (tlblX, tlblY) = midpoint(tl, bl)
            (trbrX, trbrY) = midpoint(tr, br)

            # 중간점의 유클라디안 거리 구하기
            dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
            dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

            # 길이 구하기
            if pixelsPerMetric is None:
                pixelsPerMetric = dB / 5
            dimA = round(dA / pixelsPerMetric,1)
            dimB = round(dB / pixelsPerMetric,1)
            
            # 저장
            size_list.append([dimA, dimB])
        
        try:
            result.append(size_list[1])
        except:
            result.append([-1,-1])
        
    result = np.array(result)
        
    return result


# ### color_pct(image) : 색상구하기

# In[6]:


def color_pct(images):
    
    for image in images:
        # 이미지 불러오기
        height, width = image.shape[:2]
        image = cv2.resize(image, (width, height),
                             interpolation=cv2.INTER_AREA)

        # red, green range 설정하기
        lower1 = np.array([0, 30, 30])
        upper1 = np.array([180, 255, 255])

        # HSV로 변환.
        img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # range 설정한 범위 값으로 HSV 이미지에서 마스크 생성
        img_mask = cv2.inRange(img_hsv, lower1, upper1)

        # 원본 이미지에서 마스크 범위값에 해당되는 부분 구하기
        img_result = cv2.bitwise_and(image, image, mask=img_mask)

        # result 이미지를 hsv로 변환
        img_result_hsv = cv2.cvtColor(img_result, cv2.COLOR_BGR2HSV)

        # h s v 채널로 나누기
        h, s, v = cv2.split(img_result_hsv)

        # result 이미지 중 h 채널의 전체 픽셀수
        result_cnt_ttl = (np.where(h>0,1,0).sum())
        result_cnt_ttl2 = (np.where(v>0,1,0).sum())

        # h채널 중 red 픽셀 수
        result_cnt_red=(np.where((((h>0) & (h<=15)) | ((h>=165) & (h<=180))) ,1,0).sum())

         # h채널 중 green 픽셀 수
        result_cnt_green = (np.where(((h>=25) & (h<=55)) ,1,0).sum())

        # h채널 중 dark red 픽셀수
        result_cnt_darkred = (np.where(((((h>0) & (h<=15)) 
                                        | ((h>=165) & (h<=180)))
                                        &((v>0)&(v<60))) ,1,0).sum())

        result = []
        
        try:
            # red 픽셀 비율, green 픽셀 비율
            result1= (int(result_cnt_red / result_cnt_ttl * 100))
            result2= (int(result_cnt_green / result_cnt_ttl * 100))
            result3= (int(result_cnt_darkred / result_cnt_ttl * 100))

            # 고추 색 분류하기
            if ((result1>75) & (result3<5)):
                color = '빨간고추'
            elif(result2>80):
                color = '초록고추'
            elif(result1<70 & result2<70 & result3>2):
                color = '섞인고추'
            else:
                color = '섞인고추'
            result.append([color, result1, result2, result3])

        except:
            result.append([-1,-1,-1,-1])
            
    result = np.array(result)
        
    return result


# ### pepper_att(경로, 파일이름) : size(), color_pct() 이용하여 길이, 색상 구하기

# In[7]:


def pepper_att(path_dir,name_file):
    
    images = load_images_cv(path_dir,name_file)
    
    length = size(images)
    width = length.min(axis=1)
    height = length.max(axis=1)
    
    width = width.reshape(-1,1)
    height = height.reshape(-1,1)
    
    color_group = color_pct(images)
    color = color_group[:,0]
    red = color_group[:,1]
    green = color_group[:,2]
    darkred = color_group[:,3]
    
    name_file = np.array(name_file).reshape(-1,1)
    
    info = np.concatenate((width, height, color_group, name_file), axis=1)
    
    return info


# ### pred(경로, 파일이름, model1, model2) : 이진분류모델 predict 결과 구하기

# In[21]:


def pred(default_path, file_names, model1, model2):

    images_arr = load_images(default_path, file_names)

    images_resize = resize_image(images_arr,(224,224))

    predict1 = model1.predict_classes(images_resize)
    predict2 = model2.predict_classes(images_resize)

    file_names = np.array(file_names)
    file_names = file_names.reshape(-1,1)
    data = np.concatenate((predict1, predict2), axis=1)

    return data


# ### group(array_result): 등급 분류()
# - predict1 : 0  -> 폐기
# - predict1 : 1 & predict2 : 0 -> 보통    
# - predict1 : 1 & predict2 : 1 & color : 섞인고추 -> 보통
# - predict1 : 1 & predict2 : 1 & color : 빨간고추,초록고추 -> 특상

# In[9]:


def group(data):
    
    pre1 = data[:,0].astype(np.float32)
    pre2 = data[:,1].astype(np.float32)
    color = data[:,2]

    g = np.where(color== -1, '확인필요', '')
    g = np.where(pre1 == 0, '폐기', g)
    g = np.where((pre1 == 1) & (pre2 == 0), '보통', g)
    g = np.where((pre1 == 1) & (pre2 == 1) & (color == '섞인고추') , '보통', g)
    g = np.where((pre1 == 1) & (pre2 == 1) & ((color == '빨간고추') | (color == '초록고추')) , '특상', g)

    g = g.reshape(-1,1)

    return g  


# # 데이터 분석 시작

# In[10]:


filepath = 'images/pred/'


# In[11]:


filename = os.listdir(filepath)


# In[14]:


model1 = load_model("bad_dried_good_02_0027_0.2598_0.8799_0.4104_0.8742.hdf5")
model2 = load_model("bad_dried_01_0071_0.4553_0.8052_0.4126_0.9492.hdf5")


# In[22]:


pepper_info = pepper_att(filepath, filename)

pepper_pred = pred(filepath, filename, model1, model2)

pepper_info = np.concatenate((pepper_info, pepper_pred), axis=1)

data = pepper_info[:,[7,8,2]]
grade = group(data)

pepper_info = np.concatenate((pepper_info, grade), axis=1)


# In[23]:


col_names = ['width','height','color','red','green','darkred','path','predict1','predict2','grade']
pepper_info = pd.DataFrame(pepper_info, columns = col_names)

pepper_info['datetime'] = dt.datetime.now().strftime('%y/%m/%d')
pepper_info['workplace_no'] = 1
pepper_info['no'] = pepper_info['workplace_no'].astype(str) + '_' + pepper_info.index.astype(str)

pepper_info = pd.DataFrame(pepper_info, columns = 
                           ['no','width','height','color','red','green',
                            'darkred','path', 'predict1','predict2', 'datetime','workplace_no','grade'])


# # DB 연동 후 insert

# In[ ]:


os.putenv('NLS_LANG','KOREAN_KOREA.KO16MSWIN949')

con = cx_Oracle.connect("pepper/1234@localhost:1521/xe")

pepper_info['width'] = pepper_info['width'].astype('float64')
pepper_info['height'] = pepper_info['height'].astype('float64')
pepper_info['red'] = pepper_info['red'].astype('float64')
pepper_info['green'] = pepper_info['green'].astype('float64')
pepper_info['darkred'] = pepper_info['darkred'].astype('float64')
pepper_info['predict1'] = pepper_info['predict1'].astype('float64')
pepper_info['predict2'] = pepper_info['predict2'].astype('float64')
pepper_info['workplace_no'] = pepper_info['workplace_no'].astype('object')

rows = [tuple(x) for x in pepper_info.to_records(index=False)]

cursor = con.cursor()

cursor.executemany('insert into pepper_info values (:1, :2, :3, :4, :5, :6, :7, :8, :9, :10, :11, :12, :13)', rows)

con.commit()

con.close()

