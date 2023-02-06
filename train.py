from dataset import *
from config import *
from lstm import LSTM
import torch
from torch import nn
from torch import optim
from torchnet import meter
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

writer = SummaryWriter(comment='hiddendim256layers2')
np.random.seed(0)
# 加载数据
train_dict, vocab_size, idx2word, word2idx = load_data(input_path=input_path,
                                                       output_path=output_path)
# 初始化模型
model = LSTM(vocab_size=vocab_size, hidden_dim=hidden_dim,
             embedding_dim=embedding_dim, num_layers=num_layers)
# 创建优化器
optimizer = optim.Adam(model.parameters(), lr=lr)
# 定义损失函数
criterion = nn.CrossEntropyLoss()
model.to(device)

# 计算[0]均值和[1]标准差
loss_meter = meter.AverageValueMeter()
# loss参数
best_loss = 999
# 保存对应最好准确率的模型参数
best_model = None

# 开始训练
for epoch in range(epochs):
    # 训练模型
    model.train()
    # 清零loss_meter
    loss_meter.reset()
    running_corrects = 0
    epoch_nums = 0

    for phase in ['train']:
        if phase == 'train':
            model.train()
        else:
            model.eval()

        for x, y in data_generator(train_dict):
            # x 在transpose之前的维度: (batch_size, 序列长度)
            # transpose(1,0)，将矩阵x,y轴调转。目的：将x
            # contiguous: 断开新x与旧x之间的依赖，深拷贝
            x = torch.from_numpy(x).long().transpose(1, 0).contiguous()
            x = x.to(device)
            y = torch.from_numpy(y).long().transpose(1, 0).contiguous()
            y = y.to(device)
            optimizer.zero_grad()
            with torch.set_grad_enabled(phase == 'train'):
                # 前向传播
                output_ = model(x)
                # 计算损失
                # view(-1)，将所有数调整为一维的
                # output_: (batch_size * seq_length, vocab_size)
                # y.long.view(-1): (batch_size * seq_length)
                # one-hot编码方式进行损失计算，y.long.view(-1)，的值表示了下标
                y_ = y.long().view(-1)
                loss = criterion(output_, y_)
                # 反向传播
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            # 统计
            loss_meter.add(loss.item())
            running_corrects += (output_.argmax(axis=1) == y_).sum().item()
            # print(output_.size())
            # print(y_.size())
            epoch_nums += y_.size()[0]

        epoch_acc = running_corrects / epoch_nums
        # 打印
        print("[Epoch: ] %s" % str(epoch + 1))
        print("Accuracy: %s" % str(epoch_acc))
        print("Loss: %s" % (str(loss_meter.mean)))
        writer.add_scalar('Accuracy', epoch_acc, epoch + 1)
        writer.add_scalar('Loss', loss_meter.mean, epoch + 1)

        # 更新最优模型
        if loss_meter.mean < best_loss:
            best_loss = loss_meter.mean
            best_model = model.state_dict()

        # 在训练结束时保留最优参数模型
        if epoch == epochs - 1:
            torch.save(best_model, './best_model.pkl')
writer.close()
print("finished")
