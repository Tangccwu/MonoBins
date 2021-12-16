import torch
# print(torch.hub.list('pytorch/vision'))
model = torch.hub.load('pytorch/vision:v0.4.2', 'deeplabv3_resnet101', pretrained=True)
model.eval() 

print(torch.hub.list('pytorch/vision:v0.4.2'))