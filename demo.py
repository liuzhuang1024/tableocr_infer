# -*- coding: utf-8 -*-  
from __future__ import print_function
from torch.utils.data import DataLoader
from collections import defaultdict, Counter
import argparse
import random
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import utils_1
import crnn_resnet as crnn
import re,os
import params_test
#from dataset_v3 import baiduDataset
import cv2
import time
from PIL import ImageFont,ImageDraw,Image
os.environ["CUDA_VISIBLE_DEVICES"] = "0"                                                              
import numpy as np
import copy
import pdb

def trans(input,markinlued = True):
    def getresult(input,alphabet):
        output = ""
        for char in input:
            if char in alphabet:
                output += char
            else:
                output += "-"
        return output

    if markinlued:
        output = getresult(input,params_test.alphabet+"！?；（）：,")
    else:
        alphabet = re.sub('[!？;():]', '', params_test.alphabet)
        output = getresult(input,"".join(alphabet))
    return output

def getpostion(predstabel,maxpred):
    maxpred = maxpred.cpu().numpy()
    candidate = []
    for i,index in enumerate(maxpred):
        if index != 0:
            candidate.append((index,i))
    topktabel = []
    for num, p in enumerate(predstabel):
        prob,topk = torch.topk(p, 5)
        topk = topk.cpu().numpy().tolist()
        #print(topk[0])
        topktabel.append(topk[0])
        #print("topk",topk,num)

    postionSet = []
    for index,i in candidate:
        left_pos = i
        right_pos = i
        for pt in range(i,0,-1):
            #print("left",index[0],topktabel[pt],pt:q
            if index == 3763:
                temp = topktabel[pt][:1]
            else:
                temp = topktabel[pt]
            if index[0] in temp:
                left_pos = pt
            else:
                break

        for pt in range(i,len(topktabel)):

            if index == 3763:
                temp = topktabel[pt][:1]
            else:
                temp = topktabel[pt]
            if index[0] in temp:
                #print("right", index[0], topktabel[pt],pt)
                right_pos = pt
            else:
                break
        if right_pos - left_pos > 20:
            right_pos = i
            left_pos = i

            for pt in range(i, 0, -1):
                #print("left", index[0], topktabel[pt], pt)
                if index[0] in topktabel[pt][:1]:
                    left_pos = pt
                else:
                    break

            for pt in range(i, len(topktabel)):
                if index[0] in topktabel[pt][:1]:
                    #print("right", index[0], topktabel[pt], pt)
                    right_pos = pt
                else:
                    break

        #print("length  ----",len(topktabel),[left_pos,right_pos])
        postionSet.append([max(left_pos-1,0),min(right_pos+1,131)])
    return postionSet


def drawpostion(predstabel,maxpred,img):
    width,height = img.size
    image = cv2.cvtColor(np.asarray(img.convert("RGB")),cv2.COLOR_RGB2BGR)
    postionSet = getpostion(predstabel, maxpred)
    for left,right in postionSet:
        xmin,xmax = int(left*(width/132)),int(right*(width/132))
        ymin,ymax = 3,height-3
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
    img = Image.fromarray(cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
    return img

def resize_img(cvImg):
    h, w = cvImg.shape
    new_h = 32
    new_w = int(new_h * w / h)
    input_w = 1024
    if new_w <input_w :
        input_img = np.ones((new_h,input_w),dtype=np.int)*120
        begin_index =int((input_w-new_w )/2)
        cvImg = cv2.resize(cvImg,(new_w,new_h))
        input_img[:,begin_index:begin_index+new_w] =  cvImg.copy()
    else:
        input_img = cv2.resize(cvImg,(input_w,new_h))
    input_img =input_img/255.0 
    return input_img 

def model_infer(crnn,converter,cvImg):
    cvImg = resize_img(cvImg)
    print(cvImg.shape)
    image = torch.from_numpy(cvImg).type(torch.FloatTensor)
    image.sub_(params_test.mean).div_(params_test.std)
    image = image.unsqueeze(0)
    image = image.unsqueeze(0)
    image = image.to(device)
    preds_tabel = crnn(image)
    preds_tabel = preds_tabel.permute(1 , 0 , 2)   
    pro, preds = preds_tabel.max(2)
    # print(pro)
    # print(preds)
    prob_s = torch.prod(pro).cpu().numpy()
    score = 1.0
    preds = preds.transpose(1, 0).contiguous().view(-1)
    batch_size = image.size(0)
    preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
    reg = converter.decode(preds.data, preds_size.data, raw=False)
    return reg

def model_init(model_path):
  crnn = torch.jit.load(model_path)
  crnn = crnn.to(device)  
  for p in crnn.parameters():
      p.requires_grad = False
  # crnn.eval()
  return crnn

def main(crnn):
    Iteration = 0
    accuracy = val(crnn)


class Infer:
    def __init__(self) -> None:
        manualSeed=10
        random.seed(manualSeed)
        np.random.seed(manualSeed)
        torch.manual_seed(manualSeed)
        cudnn.benchmark = True
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        converter = utils_1.strLabelConverter(params_test.dict_path)
        ##########初始化################
        model_crnn = model_init(params_test.crnn)
        ##########初始化################
    def test(self, im_path):
        img =cv2.imread(fullname,0)
        reg_result = model_infer(model_crnn,converter,img)
        return reg_result
if __name__ == '__main__':
    manualSeed=10
    random.seed(manualSeed)
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    cudnn.benchmark = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    converter = utils_1.strLabelConverter(params_test.dict_path)
    ##########初始化################
    model_crnn = model_init(params_test.crnn)
    ##########初始化################

    imglist = os.listdir(params_test.test_img)
    for imgname in imglist:    
      fullname = params_test.test_img + imgname
      print(fullname)
      img =cv2.imread(fullname,0)
      reg_result = model_infer(model_crnn,converter,img)
      print("reg_result:",reg_result)
