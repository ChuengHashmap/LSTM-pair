import codecs
import numpy as np
from config import *


# 数据读取和切分
def read_data(file_path):
    lines = codecs.open(file_path, encoding='utf-8').readlines()
    # strip: 删除字符串前后的多余空格, split(' '):转化为list，以' '为分割
    txt = [line.strip().split(' ') for line in lines]
    # 过滤掉过长的字符串
    out = [line for line in txt if (len(line) < 16 and len(line) > 3)]
    # print('read_data.out: ', out.shape)
    return out


# 加载文本数据
def load_data(input_path, output_path):
    """
    :param input_path:
    :param output_path:
    :returns train_dict:
             vocab_size:
             idx2word:
             word2idx:
    """

    # 产生数据字典
    # x 为上联，y 为下联
    # result_dict 为不同长度对联的索引
    def generate_count_dict(result_dict, x, y):
        for i, idx in enumerate(x):
            j = len(idx)
            if j not in result_dict:
                result_dict[j] = [[], []]
            result_dict[j][0].append(idx)
            result_dict[j][1].append(y[i])
        return result_dict

    # 将字典转化为numpy
    def to_numpy_array(dict):
        for count, [x, y] in dict.items():
            dict[count][0] = np.array(x)
            dict[count][1] = np.array(y)
        return dict

    x = read_data(input_path)
    y = read_data(output_path)

    # 全部词表
    vocabulary = x + y
    # 构造字符级别的特征
    string = ''
    for words in vocabulary:
        for word in words:
            string += word
    # print(string)
    # 所有词汇表,所有字的集合(去除了重复的字)
    vocabulary = set(string)

    # 构建索引
    word2idx = {word: i for i, word in enumerate(vocabulary)}
    idx2word = {i: word for i, word in enumerate(vocabulary)}

    # 词的个数,词汇表大小
    vocab_size = len(word2idx.keys())

    # 将x和y转化为索引
    x = [[word2idx[word] for word in sent] for sent in x]
    y = [[word2idx[word] for word in sent] for sent in y]

    train_dict = {}
    train_dict = generate_count_dict(train_dict, x, y)
    train_dict = to_numpy_array(train_dict)

    return train_dict, vocab_size, idx2word, word2idx


# 生成训练数据
def data_generator(data):
    """
    :param data: train_dict-{ 1 : [长度为1的上联，长度为1的下联], 2:   [长度为2的上联，长度为2的下联], 3: ...}
    :return: (batch_size, seq_len),(batch_size, seq_len, 1)
    """
    # 计算每个长度对联相应的权重
    # 统计各长度对应的对联个数
    data_nums = [float(len(x)) for num, [x, y] in data.items()]
    # 各长度字数的权重
    # print("data_nums: ", data_nums)
    data_probability = np.array(data_nums) / sum(data_nums)
    # print("data_pro", data_probability)
    # 随机选择长度不同的对联，生成batch
    for idx in range(4, 15):
        # idx = np.random.choice([i for i in range(12)], p=data_probability) + 3
        # print(idx)

        # batch_size 与 长为x_len的对联数量 之间的最小值
        x_num = len(data[idx][0])
        size = min(batch_size, x_num)

        # 随机选择size个在[0,x_num)区间的数
        idxs = np.random.choice(x_num, size=size, replace=False)
        # print("idxs: ", idxs)
        # 返回选出的上联x 与 下联y，将原本array拓展为(row,col,1)
        # expand_dims: 拓展维度
        yield data[idx][0][idxs], np.expand_dims(data[idx][1][idxs], axis=2)


if __name__ == '__main__':
    # train_dict, vocab_size, _, _ = load_data(input_path, output_path)
    # print("dict[0]", len(train_dict[3]))
    lists = read_data(input_path)
    # print("list[0]", list[0])
    dict_len = {}
    for sen in lists:
        dict_len[len(sen)] = dict_len.get(len(sen), 0) + 1
    print(dict_len)
