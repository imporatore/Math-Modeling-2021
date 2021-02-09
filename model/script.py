import time

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

from model.data_pipeline import trainLoader, testLoader
from model.image_classifier import model
from model.train import train, evaluate
from model.losses import FocalLoss
from model.config import EPOCHS, LR

from tensorboardX import SummaryWriter
writer = SummaryWriter()

# criterion = nn.BCEWithLogitsLoss()
# criterion = FocalLoss(class_num=2, alpha=torch.tensor(0.25))
# criterion = FocalLossV1()
criterion = FocalLoss()
optimizer = Adam(model.parameters(), lr=LR)

lambda_ = lambda epoch: 0.9 ** epoch
scheduler = LambdaLR(optimizer, lr_lambda=lambda_)


if __name__ == "__main__":
    train_loss = valid_loss = accuracy = []
    best_acc = None
    total_start_time = time.time()

    print('-' * 90)
    for epoch in range(1, EPOCHS+1):
        epoch_start_time = time.time()
        loss = train(model, trainLoader, criterion, optimizer)
        train_loss.append(loss*1000.)
        # writer.add_scalar('data/trainloss', train_loss, epoch)

        print('| start of epoch {:3d} | time: {:2.2f}s | loss {:5.6f}'.format(epoch,
                                                                              time.time() - epoch_start_time,
                                                                              loss))
        evaluate_start_time = time.time()
        loss, metrics = evaluate(model, testLoader, criterion)
        tp, fn, fp, tn = metrics['tp'].double(), metrics['fn'].double(), metrics['fp'].double(), metrics['tn'].double()
        corrects = tp + tn
        size = tp + fn + fp + tn
        acc = corrects / size
        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
        valid_loss.append(loss*1000.)
        # writer.add_scalar('data/validloss', valid_loss, epoch)
        accuracy.append(acc)

        print('-' * 10)
        print((time.time() - evaluate_start_time)/400)
        print('| end of epoch {:3d} | time: {:2.2f}s | loss {:.4f} | accuracy {}%({}/{}| tpr {:.4f} | fpr {:.4f} )'.format
              (epoch, time.time() - epoch_start_time, loss, acc, corrects, size, tpr, fpr))
        print('-' * 10)

        scheduler.step(epoch)
