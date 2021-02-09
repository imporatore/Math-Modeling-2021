def tp(pred, label):
    return ((pred.data == 1) & (label.data == 1)).sum()


def fn(pred, label):
    return ((pred.data == 0) & (label.data == 1)).sum()


def fp(pred, label):
    return ((pred.data == 1) & (label.data == 0)).sum()


def tn(pred, label):
    return ((pred.data == 0) & (label.data == 0)).sum()
