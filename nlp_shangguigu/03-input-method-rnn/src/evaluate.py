import torch
import config
from model import InputMethodModel
from dataset import get_dataloader
from predict import predict_batch
from tokenizer import JiebaTokenizer

def run_evaluate():
    # 1. 确定设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. 词表
    tokenizer = JiebaTokenizer.from_vacab(config.MODELS_DIR / 'vocab.txt')

    # 3. 模型
    model = InputMethodModel(vocab_size=tokenizer.vocab_size).to(device)
    model.load_state_dict(torch.load(config.MODELS_DIR / 'best_model.pth'))
    print("模型加载成功")

    # 4. 数据集
    test_dataloader = get_dataloader(train=False)

    # 5. 评估逻辑
    top1_acc, top5_acc = evaluate(model, test_dataloader, device)
    print("评估结果")
    print(f"top1: {top1_acc * 100:.2f}%")
    print(f"top5: {top5_acc * 100:.2f}%")


def evaluate(model, test_dataloader, device):
    top1_acc_count = 0
    top5_acc_count = 0
    total_count = 0

    for inputs, targets in test_dataloader:
        inputs, targets = inputs.to(device), targets.tolist()
        # inputs.shape [batch_size, seq_len]
        # targets.shape [batch_size]
        top5_index_list =  predict_batch(model, inputs)
        # top5_index_list.shape [batch_size, 5]

        for target, top5_index in zip(targets, top5_index_list):
            if target == top5_index[0]:
                top1_acc_count += 1
            if target in top5_index:
                top5_acc_count += 1
            total_count += 1
    return top1_acc_count / total_count, top5_acc_count / total_count



if __name__ == '__main__':
    run_evaluate()