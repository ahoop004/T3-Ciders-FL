import torch

def fgsm(model, criterion, images, labels, step_size):
    images = images.clone().detach().requires_grad_(True)
    outputs = model(images)
    loss = criterion(outputs, labels)
    model.zero_grad()
    loss.backward()
    adv_images = images + step_size * images.grad.sign()
    return torch.clamp(adv_images, 0, 1)

def rand_noise_attack(images, step_size):
    perturb = torch.randn_like(images)
    adv_images = images + step_size * perturb.sign()
    return torch.clamp(adv_images, 0, 1)

def pgd_attack(model, criterion, images, labels,
               eps=0.3, step_size=0.004, iters=40):
    ori = images.clone().detach()
    adv = ori.clone().detach()
    for _ in range(iters):
        adv.requires_grad_(True)
        outputs = model(adv)
        loss = criterion(outputs, labels)
        model.zero_grad()
        loss.backward()
        adv = adv + step_size * adv.grad.sign()
        eta = torch.clamp(adv - ori, min=-eps, max=eps)
        adv = torch.clamp(ori + eta, 0, 1).detach()
    return adv
