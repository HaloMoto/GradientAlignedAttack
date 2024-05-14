import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
import requests

from attack.attack import Attack

from loss.GradientAlignedLoss import GradientAlignedLoss

def query_label(images, img_id, index, api_key, api_secret, logger):
    unloader = transforms.ToPILImage()
    img = images.squeeze(0)
    img = unloader(img)
    img_path = '../images_imagga/ga_mifgsm_img_' + str(img_id) + '_' + index + '.jpg'
    img.save(img_path)
    response = requests.post(
        'https://api.imagga.com/v2/tags',
        auth=(api_key, api_secret),
        files={'image': open(img_path, 'rb')})
    response = response.json()
    logger.info(response)
    tags = response['result']['tags']

    return tags[0]['tag']['en']

class GAA_to_IMAGGA(Attack):
    r"""
    GAA_to_IMAGGA in the paper 'Boosting Adversarial Attacks with Momentum'
    [https://arxiv.org/abs/1710.06081]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        decay (float): momentum factor. (Default: 1.0)
        steps (int): number of iterations. (Default: 10)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.GAA_to_IMAGGA(model, eps=8/255, steps=10, decay=1.0, loss_type='CE', norm='linf', victim_model=None, max_queries=1)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model_list, eps=8/255, alpha=2/255, steps=10, decay=1.0, loss_type='CE', norm='linf', victim_model=None, max_queries=1, api_key=None, api_secret=None, label_projection=None):
        super().__init__("GAA_to_IMAGGA", model_list)
        self.model_list = model_list
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        self._supported_mode = ['default', 'targeted']
        self.loss_type = loss_type
        self.norm = norm
        self.victim_model = victim_model

        self.api_key = api_key
        self.api_secret = api_secret
        self.label_projection = label_projection
        self.initial_label = None
        self.final_label = None

        max_queries = max_queries-1

        if self.victim_model != None:
            assert max_queries > 0
            if max_queries > self.steps:
                max_queries = self.steps
            self.query_position = [int(np.floor(self.steps*i/max_queries)) for i in range(max_queries)]

    def forward(self, images, labels, img_id, logger):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        self.initial_label = query_label(images, img_id, 'initial', self.api_key, self.api_secret, logger)

        if self._targeted:
            target_labels = self._get_target_label(images, labels)

        momentum = torch.zeros_like(images).detach().to(self.device)

        if self.loss_type == 'CE':
            loss = nn.CrossEntropyLoss()
        elif self.loss_type == 'GAL':
            loss = GradientAlignedLoss()

        adv_images = images.clone().detach()

        outputs_victim = None
        for _ in range(self.steps):
            adv_images.requires_grad = True

            # Calculate loss
            # 受害者模型查询
            if self.victim_model != None:
                if _ in self.query_position:
                    unloader = transforms.ToPILImage()
                    img = adv_images.squeeze(0)
                    img = unloader(img)
                    img_path = '../images_imagga/ga_mifgsm_img_' + str(img_id) + '_' + str(_) + '.jpg'
                    img.save(img_path)
                    response = requests.post(
                        'https://api.imagga.com/v2/tags',
                        auth=(self.api_key, self.api_secret),
                        files={'image': open(img_path, 'rb')})
                    response = response.json()
                    logger.info(response)
                    tags = response['result']['tags']
                    self.final_label = tags[0]['tag']['en']
                    if self.initial_label != self.final_label:
                        success = True
                        return success, adv_images, int(
                                np.ceil(_ * len(self.query_position) / self.steps)) + 1, (adv_images-images).norm().item(), (adv_images-images).norm(p=np.inf).item()
                    outputs_victim = np.zeros((1,1000))
                    for index_i in range(1000):
                        for index_j in range(len(tags)):
                            tag = tags[index_j]['tag']['en']
                            confidence = tags[index_j]['confidence']
                            if tag in self.label_projection[index_i]:
                                outputs_victim[0, index_i] = confidence * self.label_projection[index_i][tag]
                    outputs_victim = torch.from_numpy(outputs_victim).to(self.device)

            if self.loss_type == 'CE':
                if self._targeted:
                    cost = -loss(self.model_list[0](adv_images), target_labels)
                    for i in range(1, len(self.model_list)):
                        cost -= loss(self.model_list[i](adv_images), target_labels)
                    cost = cost / len(self.model_list)
                else:
                    cost = loss(self.model_list[0](adv_images), labels)
                    for i in range(1, len(self.model_list)):
                        cost += loss(self.model_list[i](adv_images), labels)
                    cost = cost / len(self.model_list)
            elif self.loss_type == 'GAL':
                if self._targeted:
                    cost = -loss(self.model_list[0](adv_images), target_labels, outputs_victim, mode='score', is_logits=False)
                    for i in range(1, len(self.model_list)):
                        cost -= loss(self.model_list[i](adv_images), target_labels, outputs_victim, mode='score', is_logits=False)
                    cost = cost / len(self.model_list)
                else:
                    cost = loss(self.model_list[0](adv_images), labels, outputs_victim, mode='score', is_logits=False)
                    for i in range(1, len(self.model_list)):
                        cost += loss(self.model_list[i](adv_images), labels, outputs_victim, mode='score', is_logits=False)
                    cost = cost / len(self.model_list)

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            if self.norm == 'linf':
                grad = grad / torch.mean(torch.abs(grad),
                                         dim=(1, 2, 3), keepdim=True)
                grad = grad + momentum*self.decay
                momentum = grad

                adv_images = adv_images.detach() + self.alpha*grad.sign()
                delta = torch.clamp(adv_images - images,
                                    min=-self.eps, max=self.eps)
                adv_images = torch.clamp(images + delta, min=0, max=1).detach()
            elif self.norm == 'l2':
                grad = grad / torch.sqrt(torch.sum(torch.square(grad), dim=(1, 2, 3), keepdim=True))
                grad = grad + momentum*self.decay
                momentum = grad

                adv_images = adv_images + self.alpha*grad
                _norm = torch.sqrt(torch.sum(torch.square(adv_images-images), dim=(1, 2, 3), keepdim=True))
                factor = torch.minimum(torch.tensor(1), self.eps/_norm)
                adv_images = images + (adv_images - images) * factor
                adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        # 最后查询是否攻击成功
        if self.victim_model != None:
            self.final_label = query_label(adv_images, img_id, 'final', self.api_key, self.api_secret, logger)
            if self.initial_label != self.final_label:
                success = True
                return success, adv_images, len(self.query_position) + 1, (adv_images - images).norm().item(), (
                                   adv_images - images).norm(p=np.inf).item()
            else:
                success = False
                return success, adv_images, len(self.query_position) + 1, (adv_images - images).norm().item(), (
                                   adv_images - images).norm(p=np.inf).item()
