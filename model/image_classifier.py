from torch import nn
from torchvision import models


model = models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Sequential(
    nn.Dropout(p=0.2),
    nn.Linear(512, 2)
)
model = model.cuda()
