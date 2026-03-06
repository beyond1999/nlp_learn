import jieba
import torch
import config
from model import InputMethodModel


def predict(text):
    # 1. 确定设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # 2. 词表
    with open(config.MODELS_DIR / 'vocab.txt', 'r', encoding='utf-8') as f:
        # vocab_list = [line.strip() for line in f.readlines()]
        vocab = f.read().splitlines()

    word2index = {word: index for index, word in enumerate(vocab)}
    index2word = {index: word for index, word in enumerate(vocab)}

    # 3. 模型
    model = InputMethodModel(vocab_size=len(vocab)).to(device)
    model.load_state_dict(torch.load(config.MODELS_DIR / 'best_model.pth'))
    # 4. 处理输入
    tokens = jieba.lcut(text)
    indexes = [word2index.get(token, 0) for token in tokens]
    input_tensor = torch.tensor([indexes],dtype=torch.long).to(device)

    # 5. 预测逻辑
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        # output.shape: [batch_size, vocab_size]

    top5_indexes = torch.topk(output, k=5).indices
    # top5_indexes.shape.shape: [batch_size, 5]

    top5_indexes_list = top5_indexes.tolist()
    # 二维张量tolist会变成二维列表

    top5_tokens = [index2word[index] for index in top5_indexes_list[0]]
    return top5_tokens


if __name__ == "__main__":
    top5_tokens = predict("我们团队")
    print(top5_tokens)