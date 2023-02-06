from torch import nn


class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        """
        :param vocab_size: 词典大小
        :param embedding_dim: 词嵌入大小
        :param hidden_dim: 隐藏层大小
        :param num_layers: 隐藏层层数
        """
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        # nn.Embedding: 随机初始化向量，词向量在N(0,1)正态分布中随机取值
        # 输入: 词典大小，嵌入向量的维度
        # 输出：[规整后句子长度, batch_size, 词向量维度]
        self.embeddings = nn.Embedding(vocab_size + 1, embedding_dim)
        # lstm层
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers)
        # 线形层
        self.linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        """
        :param x: 输入批向量 (time_step(序列长度), batch_size)
        :return:
        """
        # print(x.shape)
        # x: (序列长度, batch_size)
        time_step, batch_size = x.size()
        embeds = self.embeddings(x)
        # embeds: (序列长度, batch_size, embedding_dim)
        # print("embeds dim: ", embeds.shape)
        output, (h_n, c_n) = self.lstm(embeds)
        # output: (序列长度, batch_size, hidden_dim)
        # print("output: ", output.shape)
        # 返回所有时间点的数据，每个时间点对应一个字，也就是vocab_size维度的向量
        # print("output_before:", output.reshape(time_step * batch_size, -1).shape)
        # output.reshape(time_step * batch_size,-1): (batch_size * time_step, hidden_dim)
        output = self.linear(output.reshape(time_step * batch_size, -1))
        # linear_output: (batch_size * time_step, vocab_size)
        # print("output_after: ", output.shape)
        return output


from config import *
from dataset import *
from tensorboardX import SummaryWriter
import torch

if __name__ == '__main__':
    train_dict, vocab_size, idx2word, word2idx = load_data(input_path=input_path,
                                                           output_path=output_path)
    model = LSTM(vocab_size=vocab_size, hidden_dim=hidden_dim,
                 embedding_dim=embedding_dim, num_layers=num_layers)
    writer = SummaryWriter('./runs/model_graph', comment='model1')
    fake_input = torch.randint(0, 8000, (5, 512))
    writer.add_graph(model, fake_input)
