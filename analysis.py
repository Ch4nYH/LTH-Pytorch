import torch
import matplotlib.pyplot as plt
import os
import torch.nn as nn
m1 = torch.load("mnist_1/initial_state_dict_lt.pth.tar", map_location="cpu").state_dict()

def train(model, train_loader, optimizer, criterion):
    EPS = 1e-6
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    for i in range(10):
        (imgs, targets) = next(iter(train_loader))
        optimizer.zero_grad()
        #imgs, targets = next(train_loader)
        imgs, targets = imgs.to(device), targets.to(device)
        output = model(imgs)
        train_loss = criterion(output, targets)
        train_loss.backward()
        # Freezing Pruned weights by making their gradients Zero
        for name, p in model.named_parameters():
            if 'weight' in name:
                tensor = p.data.cpu().numpy()
                grad_tensor = p.grad.data.cpu().numpy()
                grad_tensor = np.where(np.abs(tensor) < EPS, 0, grad_tensor)
                p.grad.data = torch.from_numpy(grad_tensor).to(device)
        optimizer.step()
        return train_loss.item()


import copy

for i in range(10):
    m2 = torch.load("mnist_1/{}_model_lt.pth.tar".format(i), map_location="cpu").state_dict()
    total = 0
    zeros = 0
    total_weight = 0
    pruned_weight = 0
    for key in m1.keys():
        if 'weight' in key:
            diff = m1[key] - m2[key]
            total += torch.sum(m2[key] != 0).item()
            zeros += torch.sum(torch.abs(diff[m2[key] != 0]) < 1e-9).item()
            total_weight += m2[key].numel()
            pruned_weight += torch.sum(m2[key] == 0).item()

    print(i)
    print(zeros)
    print(total)
    print(total_weight)
    print(pruned_weight)

def train(model, train_loader, optimizer, criterion):
    EPS = 1e-6
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    for i in range(20):
        (imgs, targets) = next(iter(train_loader))
        optimizer.zero_grad()
        #imgs, targets = next(train_loader)
        imgs, targets = imgs.to(device), targets.to(device)
        output = model(imgs)
        train_loss = criterion(output, targets)
        train_loss.backward()
        optimizer.step()
    return train_loss.item()

def pruning_generate(model, state_dict):
    parameters_to_prune =[]
    for (name, m) in model.named_modules():
        if isinstance(m, nn.Linear):
            m = prune.custom_from_mask(m, name = 'weight', mask = (state_dict[name + '.weight_mask'] != 0))
  
def see_remain_rate(model):
    sum_list = 0
    zero_sum = 0
    for m in model.modules():
        if isinstance(m, nn.Linear):
            sum_list = sum_list+float(m.weight.nelement())
            zero_sum = zero_sum+float(torch.sum(m.weight == 0))     
    print('remain weight = ', 100*(1-zero_sum/sum_list),'%')


def pruning_generate_percent(model,px,method='l1'):
    parameters_to_prune =[]
    for m in model.modules():
        if isinstance(m, nn.Linear):
            parameters_to_prune.append((m,'weight'))
    parameters_to_prune = tuple(parameters_to_prune)
    if method=='l1':
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=px,
        )
    elif method=='random':
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.RandomUnstructured,
            amount=px,
        )

i = 1
m4 = torch.load("mnist_1/mnist_random/{}_model_lt.pth.tar".format(i), map_location="cpu")
m3 = copy.deepcopy(m4.state_dict())   
#pruning_generate_percent(m4, 0.99, method="random")
see_remain_rate(m4)
optimizer = torch.optim.Adam(m4.parameters(), lr=0.1)
train(m4, train_loader, optimizer, criterion)
m4 = m4.state_dict()
for key in m3.keys():
    if 'orig' in key:
        weight_copy = copy.deepcopy(m3[key])
        diff = m4[key] - m3[key]
        weight_copy[(diff != 0) * (m3[key[:-5] + "_mask"] != 0)] = 0
        weight_copy[(diff == 0) * (m3[key[:-5] + "_mask"] != 0)] = 1
       
        weight_copy[m3[key[:-5] + "_mask"] == 0] = 0
        print("{} ({}%)".format(int(torch.sum(weight_copy).item()), round(int(torch.sum(weight_copy).item()) * 100.0 /  int(torch.sum((m3[key[:-5] + "_mask"] != 0)).item())  )))
        plt.figure()
        plt.imshow(weight_copy.numpy())
        if not os.path.exists("vis_3/{}".format(key)):
            os.mkdir("vis_3/{}".format(key))
        plt.savefig("vis_3/{}/{}.png".format(key,i))
        plt.figure()
        plt.imshow(m3[key[:-5] + "_mask"].numpy())
        plt.savefig("vis_3/{}/{}_mask.png".format(key,i))

for i in range(10):
    model2 = torch.load("mnist_1/{}_model_lt.pth.tar".format(i), map_location="cpu")
    m2 = copy.deepcopy(model2.state_dict())
    optimizer = torch.optim.Adam(model2.parameters(), lr=1)
    criterion = nn.CrossEntropyLoss() # Default was F.nll_loss
    import torchvision
    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
    traindataset = datasets.MNIST('~/data', train=True, download=True,transform=transform)
    testdataset = datasets.MNIST('~/data', train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(traindataset, batch_size=4, num_workers=0,drop_last=False)
    train(model2, train_loader, optimizer, criterion)
    m3 = copy.deepcopy(model2.state_dict())
    for key in m2.keys():
        if 'weight' in key:
            weight_copy = m2[key]
            diff = m3[key] - m2[key]
            weight_copy[(diff == 0) * (m2[key] == 0)] = 0
            weight_copy[(diff == 0) * (m2[key] != 0)] = 1
            weight_copy[m2[key] != 0] = 0
            print(weight_copy.shape)
            plt.figure()
            plt.imshow(weight_copy.numpy())
            if not os.path.exists("vis_2/{}".format(key)):
                os.mkdir("vis_2/{}".format(key))
            plt.savefig("vis_2/{}/{}.png".format(key,i))


    
def see_remain_rate_orig(model):
    sum_list = 0
    zero_sum = 0
    for m in model.modules():
        if isinstance(m, nn.Linear):
            sum_list = sum_list+float(m.weight_mask.nelement())
            zero_sum = zero_sum+float(torch.sum(m.weight_mask == 0))     
    print('remain weight = ', 100*(1-zero_sum/sum_list),'%')

