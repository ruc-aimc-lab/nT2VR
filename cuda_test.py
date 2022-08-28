# -*- encoding: utf-8 -*-
import torch
import torch.nn as nn
import torch.utils.data as data
import os
import math
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'

class TransformNet(nn.Module):
    """
    加入 BatchNorm, activation, dropout
    """

    def __init__(self):
        super().__init__()
        self.iters = 0
        self.a = torch.nn.Sequential(
    torch.nn.Linear(3, 1),
    torch.nn.Flatten(0, 1)
)

    def forward(self, input_x):
        self.iters = 1 + self.iters
        return self.a(input_x)


class MyDataset(data.Dataset):
    def __init__(self, x):
        self.x = x

    def __getitem__(self, index):
        return self.x[index]

model = TransformNet()

# 1） 初始化
torch.distributed.init_process_group(backend='nccl')
# 2） 配置每个进程的gpu
local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
model.to(device)
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
model = model.module


def data_parallel(model):
    global device
    device = torch.device("cuda")
    model = nn.DataParallel(model)
    model = model.to(device)
    model = model.module
    print(model.iters)
    return model
model = data_parallel(model)

x = torch.linspace(-math.pi, math.pi, 2000).to(device)
y = torch.sin(x).to(device)
p = torch.tensor([1, 2, 3]).to(device)
xx = x.unsqueeze(-1).pow(p).to(device)


loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate = 1e-6
for t in range(20000):
    # print('model iter:', model.iters)
    # Forward pass: compute predicted y by passing x to the model. Module objects
    # override the __call__ operator so you can call them like functions. When
    # doing so you pass a Tensor of input data to the Module and it produces
    # a Tensor of output data.
    y_pred = model(xx)

    # Compute and print loss. We pass Tensors containing the predicted and true
    # values of y, and the loss function returns a Tensor containing the
    # loss.
    loss = loss_fn(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())

    # Zero the gradients before running the backward pass.
    model.zero_grad()

    # Backward pass: compute gradient of the loss with respect to all the learnable
    # parameters of the model. Internally, the parameters of each Module are stored
    # in Tensors with requires_grad=True, so this call will compute gradients for
    # all learnable parameters in the model.
    loss.backward()

    # Update the weights using gradient descent. Each parameter is a Tensor, so
    # we can access its gradients like we did before.
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad