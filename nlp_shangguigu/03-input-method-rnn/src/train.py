import torch
from torch import nn
from tqdm import tqdm

from dataset import get_dataloader
from model import InputMethodModel
import config


def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    """
    训练一个轮次
    :param model: 模型
    :param dataloader: 数据集
    :param loss_fn: 损失函数
    :param optimizer: 优化器
    :param device: 设备
    :return: 平均loss
    """
    model.train()
    total_loss = 0
    for inputs, targets in tqdm(dataloader, desc="train"):
        inputs = inputs.to(device)
        targets = targets.to(device)

        # inputs.shape: [batch_size, seq_len]
        # targets.shape: [batch_size]

        # 前向传播
        outputs = model(inputs)

        # outputs.shape: [batch_size, vocab_size]

        loss = loss_fn(outputs, targets)

        # 反向传播
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    return total_loss / len(dataloader)



def train():
    # 1. 确定设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. 加载数据集
    dataloader = get_dataloader()

    # 3. 词表
    with open(config.MODELS_DIR / 'vocab.txt', 'r', encoding='utf-8') as f:
        # vocab_list = [line.strip() for line in f.readlines()]
        vocab = f.read().splitlines()

    print(vocab[:10])
    # 4. 模型
    model = InputMethodModel(vocab_size=len(vocab)).to(device)

    # 5. 损失函数
    loss_fn = nn.CrossEntropyLoss()

    # 6. 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # 开始训练
    for epoch in range(1, 1 + config.EPOCHS):
        print('=' * 10, 'Epoch {}/{}'.format(epoch, config.EPOCHS), '=' * 10)
        # 训练一个epoch的逻辑
        loss = train_one_epoch(model, dataloader, loss_fn, optimizer, device)
        print(f'loss:{loss}')


if __name__ == '__main__':
    # print(torch.cuda.is_available())
    train()