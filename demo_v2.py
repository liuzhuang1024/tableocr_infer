# -*- coding: utf-8 -*-
from __future__ import print_function
import math
from torch.utils.data import DataLoader
import random
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import utils_1
import crnn_resnet as crnn
import re
import os
import params_test
#from dataset_v3 import baiduDataset
import cv2
import time
import json
import jsonlines
from PIL import ImageFont, ImageDraw, Image
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def trans(input, markinlued=True):
    def getresult(input, alphabet):
        output = ""
        for char in input:
            if char in alphabet:
                output += char
            else:
                output += "-"
        return output

    if markinlued:
        output = getresult(input, params_test.alphabet+"！?；（）：,")
    else:
        alphabet = re.sub('[!？;():]', '', params_test.alphabet)
        output = getresult(input, "".join(alphabet))
    return output


def getpostion(predstabel, maxpred):
    maxpred = maxpred.cpu().numpy()
    candidate = []
    for i, index in enumerate(maxpred):
        if index != 0:
            candidate.append((index, i))
    topktabel = []
    for num, p in enumerate(predstabel):
        prob, topk = torch.topk(p, 5)
        topk = topk.cpu().numpy().tolist()
        # print(topk[0])
        topktabel.append(topk[0])
        # print("topk",topk,num)

    postionSet = []
    for index, i in candidate:
        left_pos = i
        right_pos = i
        for pt in range(i, 0, -1):
            # print("left",index[0],topktabel[pt],pt:q
            if index == 3763:
                temp = topktabel[pt][:1]
            else:
                temp = topktabel[pt]
            if index[0] in temp:
                left_pos = pt
            else:
                break

        for pt in range(i, len(topktabel)):

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
        postionSet.append([max(left_pos-1, 0), min(right_pos+1, 131)])
    return postionSet


def drawpostion(predstabel, maxpred, img):
    width, height = img.size
    image = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    postionSet = getpostion(predstabel, maxpred)
    for left, right in postionSet:
        xmin, xmax = int(left*(width/132)), int(right*(width/132))
        ymin, ymax = 3, height-3
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
    img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    return img


def resize_img(cvImg):
    h, w = cvImg.shape
    new_h = 32
    new_w = int(new_h * w / h)
    input_w = 1024
    if new_w < input_w:
        input_img = np.ones((new_h, input_w), dtype=np.int)*120
        begin_index = int((input_w-new_w)/2)
        cvImg = cv2.resize(cvImg, (new_w, new_h))
        input_img[:, begin_index:begin_index+new_w] = cvImg.copy()
    else:
        input_img = cv2.resize(cvImg, (input_w, new_h))
    input_img = input_img/255.0
    return input_img


def model_infer(crnn, converter, cvImg):
    # cvImg = resize_img(cvImg)
    # print(cvImg.shape)
    image = torch.from_numpy(cvImg).type(torch.FloatTensor)
    image.sub_(params_test.mean).div_(params_test.std)
    image = image.unsqueeze(0)
    image = image.unsqueeze(0)
    image = image.to(device)
    preds_tabel = crnn(image)
    preds_tabel = preds_tabel.permute(1, 0, 2)
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




class Infer:
    def __init__(self, model_pt_path, model_dict_txt) -> None:
        start = time.time()
        manualSeed = 10
        random.seed(manualSeed)
        np.random.seed(manualSeed)
        torch.manual_seed(manualSeed)
        cudnn.benchmark = True
        self.converter = utils_1.strLabelConverter(model_dict_txt)
        ##########初始化################
        self.model_crnn = model_init(model_pt_path)
        ##########初始化################
        print("Init:", time.time()-start)

    def test(self, im_path):
        img = cv2.imread(im_path, 0)
        img = self.resize_img(img)
        shape = img.shape[:2]
        reg_result = model_infer(self.model_crnn, self.converter, img)
        decode_result = index_decode_v2(reg_result)
        # print(shape, decode_result)
        # pre = re.sub("♡", "", reg_result)
        # pre = re.sub("卐", "", pre)
        # # pre = "adasd_{1234}asdas"
        # pre = re.sub("(_{)(.*?)(})", r"\2", pre)
        # decode_result = re.sub("(\^{)(.*?)(})", r"\2", pre)
        return dict(reg_result=reg_result, name=os.path.basename(im_path), im_path=im_path, shape=shape, decode_result=decode_result)

    def resize_img(self, cvImg):
        h, w = cvImg.shape
        new_h = 16  # 32
        new_w = int(new_h * w / h)
        # input_w = 384      #1024
        input_w = math.ceil(new_w/64)*64  # 1024
        if new_w < input_w:
            input_img = np.ones((new_h, input_w), dtype=np.int)*255
            begin_index = int((input_w-new_w)/2)
            cvImg = cv2.resize(cvImg, (new_w, new_h))
            input_img[:, begin_index:begin_index+new_w] = cvImg.copy()
        else:
            input_img = cv2.resize(cvImg, (input_w, new_h))
        input_img = input_img/255.0
        return input_img

    def index_decode(self, index_encode):
        # 解码部分
        res = re.sub("@\^{(.*?)}", r"<sup>\1</sup>", index_encode)
        res = re.sub("@_{(.*?)}", r"<sub>\1</sub>", res)
        res = re.split("(.{0,1}_[_&@#]{0,5})", res)
        res = list(filter(lambda x: x, res))

        # print(res)

        def trans(data, split='_#', start='<b>', end='</b>'):
            res = data

            tmp_res = []
            while res:
                tmp = res.pop(0)
                tmp_ = []
                if split in tmp:
                    tmp_ = [start + tmp.replace(split, ""), ]
                    if res:
                        tmp = res.pop(0)
                        flag = 0
                        while split in tmp:
                            tmp_.append(tmp.replace(split, ""))
                            if res:
                                tmp = res.pop(0)
                            else:
                                flag = 1
                                break

                        tmp_[-1] += end
                        if not flag:
                            tmp_.append(tmp)
                        tmp_res += tmp_
                else:
                    tmp_res.append(tmp)
            return tmp_res

        res = trans(res, '_&', '<i>', '</i>')
        res = trans(res, '_#', '<b>', '</b>')
        res = trans(res, '_@', '<strike>', '</stirke>')

        return ''.join(res)


def test(data):
    import tqdm
    print(len(data))
    td = tqdm.tqdm(data)
    f =  jsonlines.open("result_val_1228.jsonl", "a")
    for line in td:
        f.write(infer.test(line.strip("\n")))

def index_decode_v2(index_encode):
    # 解码部分
    res = re.sub("\^{(.*?)}", r"<sup>\1</sup>", index_encode)
    res = re.sub("_{(.*?)}", r"<sub>\1</sub>", res)
    res = re.split("(.{0,1}[卐♡♀]{0,3})", res)
    res = list(filter(lambda x: x, res))


    def trans(data, split='卐', start='<b>', end='</b>'):
        res = data

        tmp_res = []
        while res:
            tmp = res.pop(0)
            tmp_ = []
            if split in tmp:
                tmp_ = [start + tmp.replace(split, ""), ]
                if res:
                    tmp = res.pop(0)
                    flag = 0
                    while split in tmp:
                        tmp_.append(tmp.replace(split, ""))
                        if res:
                            tmp = res.pop(0)
                        else:
                            flag = 1
                            break

                    tmp_[-1] += end
                    if not flag:
                        tmp_.append(tmp)
                    tmp_res += tmp_
                else:
                    tmp_res.append(start + tmp.replace(split, "") + end,)
            else:
                tmp_res.append(tmp)
        return tmp_res

    res = trans(res, '♡', '<i>', '</i>')
    res = trans(res, '卐', '<b>', '</b>')
    res = trans(res, '♀', '<strike>', '</stirke>')
    return ''.join(res)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# infer = Infer(model_dict_txt="index_dict_1207.txt", model_pt_path="model/crnn_1207_decode_卐.pt")
infer = Infer(model_dict_txt="index_dict_1207.txt", model_pt_path="/data/lz/GitHub/table_ocr/crnn_resnet/crnn_1228_all.pt")

if __name__ == '__main__':
    import time
    import pprint

    with open("text.list", "r") as f:
        data = f.readlines()

    # data = ['WX20201230-193120@2x.png', 'WX20201230-193442@2x.png', 'WX20201230-193639@2x.png', 'WX20201230-193835@2x.png']
    # data = [i.strip('\n') for i in data]
    # start = time.time()
    # pp = pprint.PrettyPrinter()
    # pp.pprint([infer.test(i) for i in data])

    # 单进程
    test(data)
