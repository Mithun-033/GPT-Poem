import torch
sd=torch.load("model_epoch_10.pt",map_location="cpu")
print(type(sd))
print(len(sd))
print(list(sd.keys())[:10])