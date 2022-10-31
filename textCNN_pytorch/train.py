import torch
import os
import torch.nn as nn
from tqdm import tqdm
import time

from model import textCNN
import sen2inds
import textCNN_data

word2ind, ind2word = sen2inds.get_worddict(r'data\WordLabel.txt')
label_w2n, label_n2w = sen2inds.read_labelFile(r'data\label.txt')

textCNN_param = {
    'vocab_size': len(word2ind),
    'embed_dim': 60,
    'class_num': len(label_w2n),
    "kernel_num": 16,
    "kernel_size": [3, 4, 5],
    "dropout": 0.5,
}
dataLoader_param = {
    'batch_size': 128,
    'shuffle': True,
}


def main():
    #init net
    print('init net...')
    net = textCNN(textCNN_param)
    weightFile = r'model\weight.pkl'
    if os.path.exists(weightFile):
        print('load weight')
        net.load_state_dict(torch.load(weightFile))
    else:
        net.init_weight()
    print(net)

    net.cuda()

    #init dataset
    print('init dataset...')
    dataLoader = textCNN_data.textCNN_dataLoader(dataLoader_param)

    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    criterion = nn.NLLLoss()

    log = open('log\log_{}.txt'.format(time.strftime('%y%m%d%H')), 'w')
    log.write('epoch step loss\n')
    log_test = open('log\log_test_{}.txt'.format(time.strftime('%y%m%d%H')), 'w')
    log_test.write('epoch step test_acc\n')
    print("training...")
    for epoch in range(100):
        bar = tqdm(enumerate(dataLoader), total=len(dataLoader), ascii=True, desc="train")
        for i, (clas, sentences) in bar:
            optimizer.zero_grad()
            sentences = sentences.type(torch.LongTensor).cuda()
            clas = clas.type(torch.LongTensor).cuda()
            out = net(sentences)
            loss = criterion(out, clas)
            loss.backward()
            optimizer.step()

            if (i + 1) % 10 == 0:
                bar.set_description("loss: %f" % loss.item())
                data = str(epoch + 1) + ' ' + str(i + 1) + ' ' + str(loss.item()) + '\n'
                log.write(data)
        torch.save(net.state_dict(), weightFile)


if __name__ == "__main__":
    main()