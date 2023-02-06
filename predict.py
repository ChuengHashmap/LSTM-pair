import torch
from dataset import *
from lstm import LSTM
from config import *

_, vocab_size, idx2word, word2idx = load_data(input_path, output_path)


def predict(s):
    # 将字符串转化为编码
    x = [word2idx[word] for word in s]
    # print("x_before: ", x)
    # x: (1,seq_length)
    # 讲list转化为tensor
    x = torch.from_numpy(np.array(x).reshape(-1, 1))
    # print("x_after: ", x)
    # x: (seq_length, 1)

    # 加载模型
    model_path = './best_model.pkl'
    model = LSTM(vocab_size=vocab_size, hidden_dim=hidden_dim,
                 embedding_dim=embedding_dim, num_layers=num_layers)

    model.load_state_dict(torch.load(model_path, 'cpu'))

    y = model(x.long())
    # print("y: ", y.shape)
    # 取每一列最大的行的行标
    # 当axis=0，是在列中比较，选出最大的 行 索引
    # 当axis=1，是在行中比较，选出最大的 列 索引
    y = y.argmax(axis=1)
    # print("y: ", y)
    r = ''.join([idx2word[idx.item()] for idx in y])

    print('上联：%s， 下联：%s' % (s, r))


sentence = '恭喜发财'
predict(sentence)
