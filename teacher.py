import torch

x = torch.arange(1, 41).view(4, 10).float() / 40
y = torch.arange(4)

W1 = torch.arange(1, 21, dtype=torch.float).view(10, 2) / 20
b1 = torch.arange(1, 3, dtype=torch.float) / 2

W2 = torch.arange(1, 9, dtype=torch.float).view(2, 4) / 8
b2 = torch.arange(1, 5, dtype=torch.float) / 4

W1.requires_grad = True
W1.retain_grad()

b1.requires_grad = True
b1.retain_grad()

W2.requires_grad = True
W2.retain_grad()

b2.requires_grad = True
b2.retain_grad()

z1 = x @ W1 + b1
z1.retain_grad()

a1 = torch.nn.functional.relu(z1)
a1.retain_grad()

z2 = a1 @ W2 + b2
z2.retain_grad()

a2 = torch.nn.functional.softmax(z2, dim=1)
a2.retain_grad()
a2_log = a2.log()
a2_log.retain_grad()

loss = torch.nn.functional.nll_loss(a2_log, y, reduction='none')
loss.retain_grad()

l = loss.sum()
l.backward()
print(loss.grad)

with torch.no_grad():
    print('--- W1 grad ---')
    print(W1.grad)
    
    print('--- b1 grad ---')
    print(b1.grad)

    print('--- W2 grad ---')
    print(W2.grad)
    
    print('--- b2 grad ---')
    print(b2.grad)

    print('--- z1 grad ---')
    print(z1.grad)

    print('--- a1 grad ---')
    print(a1.grad)

    print('--- z2 grad ---')
    print(z2.grad)

    print('--- a2 grad ---')
    print(a2.grad)
