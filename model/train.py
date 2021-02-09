import torch
from torch.autograd import Variable

from model.metrics import tp, fn, fp, tn
metrics = [tp, fn, fp, tn]


def train(model, trainloader, criterion, opt):
    model.train()
    size = len(trainloader.dataset)
    total_loss = 0

    for image, label in trainloader:
        image = Variable(image.cuda())
        label = Variable(label.cuda())
        opt.zero_grad()

        target = model(image)
        loss = criterion(target, label)
        loss.backward()
        opt.step()

        total_loss += loss.item()
    return total_loss/float(size)


def evaluate(model, testloader, criterion):
    model.eval()
    size = len(testloader.dataset)
    corrects = eval_loss = 0
    metrics_ = {f.__name__: 0 for f in metrics}
    with torch.no_grad():
        for image, label in testloader:
            image = Variable(image.cuda())
            label = Variable(label.cuda())
            pred = model(image)
            loss = criterion(pred, label)

            eval_loss += loss.item()
            for f in metrics:
                metrics_[f.__name__] += f(torch.max(pred, 1)[1].view(label.size()), label)
    return eval_loss/float(size), metrics_


if __name__ == "__main__":
    pass
