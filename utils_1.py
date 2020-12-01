#!/usr/bin/python
# encoding: utf-8

#!/usr/bin/python
# encoding: utf-8

import torch
import torch.nn as nn
from torch.autograd import Variable
import collections
# import params_test
from tqdm import tqdm
import numpy as np 
import cv2
import os
import random

class strLabelConverter(object):
    """Convert between str and label.

    NOTE:
        Insert `blank` to the alphabet for CTC.

    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, alphabet, ignore_case=False):
        self._ignore_case = ignore_case
        if self._ignore_case:
            #alphabet = alphabet.lower()
            alphabet = alphabet.lower()
        self.alphabet = alphabet  # for `-1` index
        dict_lines = open(alphabet,'r',encoding='utf-8').readlines()
        self.dict = {}
        self.worddict = {}
        #self.worddict[0] = 'blank'
        for i, char in enumerate(dict_lines):
            # # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            # #self.dict[char.strip('\n').strip('\r')] = i + 1
            # self.dict[char.strip('\n').strip('\r')] = i 
            # #self.worddict[int(i+1)] = char.strip('\n').strip('\r')
            # self.worddict[int(i)] = char.strip('\n').strip('\r')
            char = char.replace('\n','')
            # print(char)
            # print(len(char))
            # print(char[len(char)-1])
            try:
                index,latex = char.split()
            except:
                index = char
                latex = " "

            if index not in self.worddict:
                self.worddict[int(index)] = latex

        # print("self dict",self.worddict)


    def encode(self, text):
        """Support batch or single str.

        Args:
            text (str or list of str): texts to convert.

        Returns:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        """

        
        length = []
        result = []
        decode_flag = True if type(text[0])==bytes else False
        
        for item in text:
            if decode_flag:
                item = item.decode('utf-8','strict')
            length.append(len(item))
            for char in item:
                result.append(int(char))
                # if char in self.dict.keys():
                #     if char == '-' or char == '―':
                #         index = self.dict["—"]
                #     else:
                #         index = self.dict[char]
                # else:
                #     if char == '-' or char == '―':
                #         index = self.dict["—"]
                #     else:
                #         #print("text,char:",item,char)
                #         index = self.dict["&"]
                # #elif char == '-' or char == '―':
                # #    index = self.dict["—"]
                # result.append(index)
        
        # print('text:',text)
        # for item in text:
            # text = [int(s) for s in text]
        text = result
        #total_label = torch.from_numpy(np.asarray(total_label))
        #total_label = total_label.int()
        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode(self, t, length, raw=False):

        """Decode encoded texts back into strs.

        Args:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.

        Raises:
            AssertionError: when the texts and its length does not match.

        Returns:
            text (str or list of str): texts to convert.
        """
        #t = t.cpu().numpy()

        if raw=="index2latex":
            latex = []
            for one in t :
                print(one)
                one_latex = [self.worddict[int(s)] for s in one]
                latex.append(''.join(one_latex))
            return latex

            
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(), length)
            if raw:
                #return ''.join([self.alphabet[i - 1] for i in t])
                return ''.join([list(self.dict.keys())[list(self.dict.values()).index(i)] for i in t])
            else:
                char_list = []
                #print("t[]:",t)
                for i in range(length):
                    #print("t shape:",t.shape)
                    #print("i:",i)
                    #print("t[i]:",t)
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        #print("t:",t)
                        #char_list.append(self.alphabet[t[i] - 1])
                        #char_list.append(list(self.dict.keys())[list(self.dict.values()).index(t[i])]) #while train -->val
                        #print("t[i]:",self.worddict[int(t[i])])

                        char_list.append(self.worddict[int(t[i])])


                #print("".join(char_list))
                return ''.join(char_list)
        else:
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts


class averager(object):
    """Compute average for `torch.Variable` and `torch.Tensor`. """

    def __init__(self):
        self.reset()

    def add(self, v):
        if isinstance(v, Variable):
            count = v.data.numel()
            v = v.data.sum()
        elif isinstance(v, torch.Tensor):
            count = v.numel()
            v = v.sum()

        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res


def oneHot(v, v_length, nc):
    batchSize = v_length.size(0)
    maxLength = v_length.max()
    v_onehot = torch.FloatTensor(batchSize, maxLength, nc).fill_(0)
    acc = 0
    for i in range(batchSize):
        length = v_length[i]
        label = v[acc:acc + length].view(-1, 1).long()
        v_onehot[i, :length].scatter_(1, label, 1.0)
        acc += length
    return v_onehot


def loadData(v, data):
    v.data.resize_(data.size()).copy_(data)
    #print(v.size())


def prettyPrint(v):
    print('Size {0}, Type: {1}'.format(str(v.size()), v.data.type()))
    print('| Max: %f | Min: %f | Mean: %f' % (v.max().data[0], v.min().data[0],
                                              v.mean().data[0]))


def assureRatio(img):
    """Ensure imgH <= imgW."""
    b, c, h, w = img.size()
    if h > w:
        main = nn.UpsamplingBilinear2d(size=(h, h), scale_factor=None)
        img = main(img)
    return img

def to_alphabet(path):
    with open(path, 'r', encoding='utf-8') as file:
        alphabet = list(set(''.join(file.readlines())))

    return alphabet

def get_batch_label(d, i):
    
    label = []
    for idx in i:
        label.append(list(d.labels[idx].values())[0])
    return label

def compute_std_mean(txt_path, image_prefix, NUM=None):
    
    imgs = np.zeros([params_2.imgH, params_2.imgW, 1, 1])
    means, stds = [], []
    with open(txt_path, 'r') as file:
        contents = [c.strip().split(' ')[0] for c in file.readlines()]
        if NUM is None:
            NUM = len(contents)
        else:
            random.shuffle(contents)
        for i in tqdm(range(NUM)):
            file_name = contents[i]
            img_path = os.path.join(image_prefix, file_name)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            h, w = img.shape[:2]
            img = cv2.resize(img, (0,0), fx=params_2.imgW/w, fy=params_2.imgH/h, interpolation=cv2.INTER_CUBIC)
            img = img[:, :, np.newaxis, np.newaxis]
            imgs = np.concatenate((imgs, img), axis=3)
    imgs = imgs.astype(np.float32) / 255.

    for i in range(1):
        pixels = imgs[:, :, i, :].ravel()
        means.append(np.mean(pixels))
        stds.append(np.std(pixels))

    # means.reverse()  # BGR --> RGB
    # stdevs.reverse()
    # print(means, stds)

    return stds, means
