# Gradient-Aligned Attack

## Description
This is the code repository of the paper "Improving Query Efficiency of Black-Box Attacks via the Preference of Deep Learning Models".

## Datasets
- ImageNet
- Real-world API, [IMAGGA](https://imagga.com)

## Pretrained Models
The used surrogate and victim models are the pretrained models from [pytorch.org](https://pytorch.org/vision/stable/models.html) and [huggingface.co](https://huggingface.co/docs/timm/models/ensemble-adversarial).

## Running Scripts
```bash
# Running GAA on the ImageNet in the low-query scenario
python -u ./ImageNet/run_GAA.py --ensemble-surrogate1 --victim-model-type vgg19 --test-batch-size 1 --max-query 5 --gpu 0

# Running BASES with GACE on the ImageNet
python -u ./ImageNet/run_BASES_with_GACE.py --ensemble-surrogate1 --victim-model-type vgg19 --test-batch-size 1 --gpu 0

# Attacking real-world API, IMAGGA
python -u ./ImageNet/attack_IMAGGA.py --ensemble-surrogate1 --test-batch-size 1 --gpu 0
```
