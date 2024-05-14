import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torchvision import transforms
import requests

from loss.GradientAlignedLoss import GradientAlignedLoss

from attack.attack import Attack

softmax = torch.nn.Softmax(dim=1)

def get_logits_probs(im, model):
    """Get the logits of a model given an image as input
    Args:
        im (PIL.Image or np.ndarray): uint8 image (read from PNG file), for 0-1 float image
        model (torchvision.models): the model returned by function load_model
    Returns:
        logits (numpy.ndarray): direct output of the model
        probs (numpy.ndarray): logits after softmax function
    """
    logits = model(im)
    probs = softmax(logits)
    return logits, probs

def loss_cw(logits, tgt_label, margin=200, targeted=True):
    """c&w loss: targeted
    Args:
        logits (Tensor):
        tgt_label (Tensor):
    """
    device = logits.device
    k = torch.tensor(margin).float().to(device)
    tgt_label = tgt_label.squeeze()
    logits = logits.squeeze()
    onehot_logits = torch.zeros_like(logits)
    onehot_logits[tgt_label] = logits[tgt_label]
    other_logits = logits - onehot_logits
    best_other_logit = torch.max(other_logits)
    tgt_label_logit = logits[tgt_label]
    if targeted:
        loss = torch.max(best_other_logit - tgt_label_logit, -k)
    else:
        loss = torch.max(tgt_label_logit - best_other_logit, -k)
    return loss

class GACE_loss(nn.Module):

    def __init__(self, target=False):
        super(GACE_loss, self).__init__()
        self.target = target
        self.loss_gace = GradientAlignedLoss()

    def forward(self, logits, label, logits_victim, is_logits=True):
        loss = self.loss_gace(logits, label, logits_victim, is_logits=is_logits)
        if self.target:
            return loss
        else:
            return -loss

class CE_loss(nn.Module):

    def __init__(self, target=False):
        super(CE_loss, self).__init__()
        self.target = target
        self.loss_ce = torch.nn.CrossEntropyLoss()

    def forward(self, logits, label):
        loss = self.loss_ce(logits, label)
        if self.target:
            return loss
        else:
            return -loss

class CW_loss(nn.Module):

    def __init__(self, target=False):
        super(CW_loss, self).__init__()
        self.target = target
        self.loss_cw = loss_cw

    def forward(self, logits, label):
        return loss_cw(logits, label, targeted=self.target)

def get_loss_fn(loss_name, targeted=True):
    """get loss function by name
    Args:
        loss_name (str): 'cw', 'ce', 'hinge' ...
    """
    if loss_name == 'ce':
        return CE_loss(targeted)
    elif loss_name == 'gace':
        return GACE_loss(targeted)
    elif loss_name == 'cw':
        return CW_loss(targeted)

def get_label_loss(im, model, tgt_label, loss_name, targeted=True):
    """Get the loss
    Args:
        im (PIL.Image): uint8 image (read from PNG file)
        tgt_label (int): target label
        loss_name (str): 'cw', 'ce'
    """
    loss_fn = get_loss_fn(loss_name, targeted=targeted)
    logits, _ = get_logits_probs(im, model)
    pred_label = logits.argmax().detach()
    loss = loss_fn(logits, tgt_label).detach()
    return pred_label, loss, logits

def get_imagga_loss(adv, label_list_clean, img_id, index, api_key, api_secret, logger):
    unloader = transforms.ToPILImage()
    img = adv.squeeze(0)
    img = unloader(img)
    img_path = '../images_imagga/BASES_img_' + str(img_id) + '_' + str(index) + '.jpg'
    img.save(img_path)
    response = requests.post(
        'https://api.imagga.com/v2/tags',
        auth=(api_key, api_secret),
        files={'image': open(img_path, 'rb')})
    response = response.json()
    logger.info(response)
    tags = response['result']['tags']

    if label_list_clean == None:
        label_list = []
        for i in range(len(tags)):
            tag = tags[i]['tag']['en']
            label_list.append(tag)
        return label_list, response
    else:
        score_correct = 0
        score_wrong = 0
        for i in range(len(tags)):
            tag = tags[i]['tag']['en']
            confidence = tags[i]['confidence']
            if tag in label_list_clean:
                score_correct += confidence
            else:
                score_wrong += confidence
        # define loss function for imagga
        loss = score_correct - score_wrong

        def success():
            return tags[0]['tag']['en'] != label_list_clean[0]

        is_success = success()

        return loss, is_success, response

def get_adv(im, target, w, pert_machine, bound, eps, n_iters, alpha, algo='mim', fuse='loss', untargeted=False, intermediate=False, loss_name='gace', adv_init=None, logits_victim=None):
    """Get the adversarial image by attacking the perturbation machine
    Args:
        im (torch.Tensor): original image
        target (torch.Tensor): 0-999
        w (torch.Tensor): weights for models
        pert_machine (list): a list of models
        bound (str): choices=['linf','l2'], bound in linf or l2 norm ball
        eps, n_iters, alpha (float/int): perturbation budget, number of steps, step size
        algo (str): algorithm for generating perturbations
        fuse (str): methods to fuse ensemble methods. logit, prediction, loss
        untargeted (bool): if True, use untargeted attacks, target_idx is the true label.
        intermediate (bool): if True, save the perturbation at every 10 iters.
        loss_name (str): 'cw', 'ce', 'hinge' ...
    Returns:
        adv (torch.Tensor): adversarial output
    """
    # device = next(pert_machine[0].parameters()).device
    n_wb = len(pert_machine)
    if algo == 'mim':
        g = 0 # momentum
        mu = 1 # decay factor

    if adv_init is None:
        adv = torch.clone(im) # initial adversarial image
    else:
        adv = adv_init

    loss_fn = get_loss_fn(loss_name, targeted = not untargeted)
    losses = []
    if intermediate:
        adv_list = []
    for i in range(n_iters):
        adv.requires_grad =True
        outputs = [model(adv) for model in pert_machine]

        if fuse == 'loss':
            if loss_name == 'gace':
                loss = sum([w[idx] * loss_fn(outputs[idx], target, logits_victim, is_logits=False) for idx in range(n_wb)])
            else:
                loss = sum([w[idx] * loss_fn(outputs[idx], target) for idx in range(n_wb)])

        elif fuse == 'prob':
            target_onehot = F.one_hot(target, 1000)
            prob_weighted = torch.sum(torch.cat([w[idx] * softmax(outputs[idx]) for idx in range(n_wb)], 0), dim=0, keepdim=True)
            loss = - torch.log(torch.sum(target_onehot *prob_weighted))
        elif fuse == 'logit':
            logits_weighted = sum([w[idx] * outputs[idx] for idx in range(n_wb)])
            if loss_name == 'gace':
                loss = loss_fn(logits_weighted, target, logits_victim, is_logits=False)
            else:
                loss = loss_fn(logits_weighted, target)

        losses.append(loss.item())
        loss.backward()

        with torch.no_grad():
            grad = adv.grad
            if algo == 'fgm':
                # needs a huge learning rate
                adv = adv - alpha * grad / torch.norm(grad, p=2)
            elif algo == 'pgd':
                adv = adv - alpha * torch.sign(grad)
            elif algo == 'mim':
                g = mu * g + grad / torch.norm(grad, p=1)
                adv = adv - alpha * torch.sign(g)

            if bound == 'linf':
                # project to linf ball
                adv = (im + (adv - im).clamp(min=-eps ,max=eps)).clamp(0, 1)
            else:
                # project to l2 ball
                pert = adv - im
                l2_norm = torch.norm(pert, p=2)
                if l2_norm > eps:
                    pert = pert / l2_norm * eps
                adv = (im + pert).clamp(0, 1)

        if intermediate and i% 10 == 9:
            adv_list.append(adv.detach())
    if intermediate:
        return adv_list, losses
    return adv.detach(), losses

class GA_BASES_to_IMAGGA(Attack):
    def __init__(self, surrogate_model_list, n_wb, bound, eps, n_iters, algo, fuse, loss_name, alpha, lr, iterw, api_key, api_secret, label_projection):
        super(GA_BASES_to_IMAGGA, self).__init__('GA_BASES_to_IMAGGA', surrogate_model_list)
        self.surrogate_model_list = surrogate_model_list
        self.n_wb = n_wb
        self.bound = bound
        self.eps = eps
        self.n_iters = n_iters
        self.algo = algo
        self.fuse = fuse
        self.loss_name = loss_name
        self.lr = lr
        self.iterw = iterw
        self.alpha = alpha # step-size
        self.api_key = api_key
        self.api_secret = api_secret
        self.label_projection = label_projection

        self._supported_mode = ['default', 'targeted']

    def forward(self, image, label, img_id, logger):
        # query Imagga for clean images
        label_list_clean = None
        label_list_clean, response = get_imagga_loss(image, label_list_clean, img_id, 0, self.api_key, self.api_secret, logger)
        # determine the logits of victim model
        tags = response['result']['tags']
        logits_victim = np.zeros((1,1000))
        for index_i in range(1000):
            for index_j in range(len(tags)):
                tag = tags[index_j]['tag']['en']
                confidence = tags[index_j]['confidence']
                if tag in self.label_projection[index_i]:
                    logits_victim[0, index_i] = confidence * self.label_projection[index_i][tag]
        logits_victim = torch.from_numpy(logits_victim).to(self.device)

        lr_w = self.lr

        # start from equal weights
        w_np = np.array([1 for _ in range(len(self.surrogate_model_list))]) / len(self.surrogate_model_list)
        adv, losses = get_adv(image, label, w_np, self.surrogate_model_list, self.bound, self.eps, self.n_iters, self.alpha, algo=self.algo, fuse=self.fuse, untargeted=not self._targeted, loss_name='ce', logits_victim=logits_victim)
        loss, is_success, response = get_imagga_loss(adv, label_list_clean, img_id, 1, self.api_key, self.api_secret, logger)
        n_query = 1

        if (self._targeted and is_success) or (not self._targeted and is_success):
            # originally successful
            success = True
            return success, adv, n_query, (adv-image).norm().item(), (adv-image).norm(p=np.inf).item()
        else:
            idx_w = 0 # idx of wb in W
            last_idx = 0 # if no changes after one round, reduce the learning rate

            while n_query < self.iterw:
                w_np_temp_plus = w_np.copy()
                w_np_temp_plus[idx_w] += lr_w

                # determine the logits of victim model
                tags = response['result']['tags']
                logits_victim = np.zeros((1, 1000))
                for index_i in range(1000):
                    for index_j in range(len(tags)):
                        tag = tags[index_j]['tag']['en']
                        confidence = tags[index_j]['confidence']
                        if tag in self.label_projection[index_i]:
                            logits_victim[0, index_i] = confidence * self.label_projection[index_i][tag]
                logits_victim = torch.from_numpy(logits_victim).to(self.device)

                adv_plus, losses_plus = get_adv(image, label, w_np_temp_plus, self.surrogate_model_list, self.bound, self.eps, self.n_iters, self.alpha, algo=self.algo, fuse=self.fuse, untargeted=not self._targeted, loss_name=self.loss_name, adv_init=adv, logits_victim=logits_victim)
                loss_plus, is_success, response_plus = get_imagga_loss(adv_plus, label_list_clean, img_id, n_query+1, self.api_key, self.api_secret, logger)
                n_query += 1

                # stop if successful
                if self._targeted * is_success or (not self._targeted) * is_success:
                    adv = adv_plus
                    success = True
                    return success, adv, n_query, (adv-image).norm().item(), (adv-image).norm(p=np.inf).item()

                w_np_temp_minus = w_np.copy()
                w_np_temp_minus[idx_w] -= lr_w

                # determine the logits of victim model
                tags = response['result']['tags']
                logits_victim = np.zeros((1, 1000))
                for index_i in range(1000):
                    for index_j in range(len(tags)):
                        tag = tags[index_j]['tag']['en']
                        confidence = tags[index_j]['confidence']
                        if tag in self.label_projection[index_i]:
                            logits_victim[0, index_i] = confidence * self.label_projection[index_i][tag]
                logits_victim = torch.from_numpy(logits_victim).to(self.device)

                adv_minus, losses_minus = get_adv(image, label, w_np_temp_minus, self.surrogate_model_list, self.bound, self.eps, self.n_iters, self.alpha, algo=self.algo, fuse=self.fuse, untargeted=not self._targeted, loss_name=self.loss_name, adv_init=adv, logits_victim=logits_victim)
                loss_minus, is_success, response_minus = get_imagga_loss(adv_minus, label_list_clean, img_id, n_query+1, self.api_key, self.api_secret, logger)
                n_query += 1

                # stop if successful
                if self._targeted * is_success or (not self._targeted) * is_success:
                    adv = adv_minus
                    success = True
                    return success, adv, n_query, (adv-image).norm().item(), (adv-image).norm(p=np.inf).item()

                # update
                if loss_plus < loss_minus:
                    w_np = w_np_temp_plus
                    adv = adv_plus
                    last_idx = idx_w
                    response = response_plus
                elif loss_plus > loss_minus:
                    w_np = w_np_temp_minus
                    adv = adv_minus
                    last_idx = idx_w
                    response = response_minus

                idx_w = (idx_w+1)%self.n_wb
                if n_query > 5 and last_idx == idx_w:
                    lr_w /= 2

            success = False
            return success, adv, n_query, (adv-image).norm().item(), (adv-image).norm(p=np.inf).item()