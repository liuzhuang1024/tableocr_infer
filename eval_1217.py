import jsonlines
import os.path as osp
import os
import json
from string_distance.edit_distance import levenshtein
import time

class Ev:
    def __init__(self) -> None:
        self.max_length = 0
        self.max_distance = 0

        self.max_seq = 0
        self.max_right_seq = 0

    def count(self, x, y):
        cost = levenshtein(x, y)
        self.max_distance += cost
        self.max_length += max(len(x), len(y))
        if x == y:
            self.max_right_seq += 1
        self.max_seq += 1
        return cost

    def socre(self, ):
        char_acc = 1 - self.max_distance / self.max_length
        seq_acc = self.max_right_seq / self.max_seq
        return dict(char_acc=char_acc, seq_acc=seq_acc, max_seq=self.max_seq, max_right_seq=self.max_right_seq, max_distance=self.max_distance, max_length=self.max_length)


def fliter_b_i_strike(label):
    value = re.sub("<b>", "粗", label)
    value = re.sub("</b>", "细", value)
    value = re.sub("<i>", "斜", value)
    value = re.sub("</i>", "直", value)
    value = re.sub("<sub>", "上", value)
    value = re.sub("</sub>", "下", value)
    value = re.sub("<sup>", "左", value)
    label = re.sub("</sup>", "右", value)
    return label


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


if __name__ == "__main__":
    import re
    #  read result
    # reader = list(jsonlines.open("result_val_1228.jsonl", "r").iter())
    # new_reader = {}
    # for d in reader:
    #     new_reader[d['name']] = d['decode_result']
    # result_txt = "best_result"
    # print(result_txt)

    # read gt
    with open("../table_ocr/data/val.txt", "r") as f:
        data = f.readlines()
    gt_dict = {}
    for line in data:
        name, tokens = line.strip('\n').split('\t')
        gt_dict[name] = tokens

    # read result
    result_txt = "/data/lz/GitHub/vietocr/transformer.result"
    print(result_txt)
    with open(result_txt, "r") as f:
        data = f.readlines()
    new_reader = {}
    for line in data:
        # name, tokens = line.strip("\n").split(r".png ", maxsplit=1)
        name, tokens = line.strip("\n").split("\t", maxsplit=1)
        # if not tokens:
        #     print(name)
        # tokens = index_decode_v2(tokens)
        new_reader[osp.basename(name)] = tokens

    f = open(f"{result_txt}_{time.strftime('%y_%m_%d')}.csv", "w")
    f.write("name,distance,gt,predict\n")
    
    ev = Ev()
    for key, tokens in new_reader.items():
        gt_tokens = gt_dict.get(key, False)
        tokens, gt_tokens = fliter_b_i_strike(tokens),fliter_b_i_strike(gt_tokens)
        # if len(gt_tokens) < 50:
        #     continue
        if gt_tokens:
            cost = ev.count(tokens, gt_tokens)
            if cost / max(len(tokens), len(gt_tokens)) > 0.5:
                f.write(f"{key},{cost},{gt_tokens},{tokens}\n")
    print("替换", ev.socre())

    ev = Ev()
    for key, tokens in new_reader.items():
        gt_tokens = gt_dict.get(key, False)
        # tokens, gt_tokens = fliter_b_i_strike(tokens),fliter_b_i_strike(gt_tokens)
        # if len(gt_tokens) < 50:
        #     continue
        if gt_tokens:
            cost = ev.count(tokens, gt_tokens)
            if cost / max(len(tokens), len(gt_tokens)) > 0.5:
                f.write(f"{key},{cost},{gt_tokens},{tokens}\n")
    print("非替换", ev.socre())