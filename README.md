# Masking implementation for Unifying Language Learning Paradigms (UL2)

Want to get better model with limited budgets? You are in the right place

<p align="center">
  <img src="https://github.com/theblackcat102/unify-learning-paradigms/blob/master/ul2.png?raw=true" width="600">
</p>

- R-Denoiser (μ=3,r=0.15,n)∪ (μ=8,r=0.15,n)

    The regular denoising is the standard span corruption introduced in Raffel et al. (2019) that uses a range of 2 to 5 tokens as the span length, which masks about 15% of input tokens

- S-Denoiser (μ=L/4,r=0.25,1)

    A specific case of denoising where we observe a strict sequential order when framing the inputs-to-targets task, i.e., prefix language modeling

- X-Denoiser (μ = 3,r = 0.5,n)∪(μ = 8,r = 0.5,n)∪(μ = 64,r =0.15,n)∪ (μ=64,r=0.5,n)

    An extreme version of denoising where the model must recover a large part of the input, given a small to moderate part of it. This simulates a situation where a model needs to generate long target from a memory with relatively limited information. To do so, we opt to include examples with aggressive denoising where approximately 50% of the input sequence is masked

2022 papers : Transcending Scaling Laws with 0.1% Extra Compute

>  we show an approximately 2x computational savings rate

- Regular denoising whereby the noise is sampled as spans, replaced with sentinel tokens. This is also the standard span corruption task used in Raffel et al. (2019). Spans are typically uniformly sampled with a mean of 3 and a corruption rate of 15%.

- Extreme denoising whereby the noise is increased to relatively ‘extreme‘ amounts in either a huge percentage of the original text or being very long in nature. Spans are typically uniformly sampled with a mean length of 32 OR a corruption rate of up to 50%.

- Sequential denoising whereby the noise is always sampled from the start of the text to a randomly sampled point in the text. This is also known as the PrefixLM objective (not to be confused with the architecture).

This repo will just aim for accompolish this task instead, UL2 is way too complicated for my likings

> 50% PrefixLM, 25% Long (extreme) span corruption, and 25% regular span corruption to be quite simple and efficient


## Experiments

Run a mT5 encoder pretraining on 3090 on pythia json.zst files

```
python pretrain_example.py
```

<p align="center">
  <img src="https://github.com/theblackcat102/theblackcat102.github.io/raw/master/images/ul2_loss_func.png" width="600">
</p>

training loss was stable and no weird spikes

## References

Core Papers

[Transcending Scaling Laws with 0.1% Extra Compute](https://arxiv.org/pdf/2210.11399.pdf)

[Unifying Language Learning Paradigms](https://arxiv.org/pdf/2205.05131.pdf)

Implements of t5 noise masking in huggingface transformers or python code

[OSLO](https://github.com/EleutherAI/oslo) : very underrated, some tidy and documentation, this will be a very useful tool

 - [t5_pretraining.py](https://github.com/EleutherAI/oslo/blob/main/oslo/transformers/tasks/data_t5_pretraining.py)
    
    Heavily inspired from this section

[Amazon science : label aware pretrain in python](https://github.com/amazon-science/label-aware-pretrain/blob/main/models/preprocessor.py)

[Fairseq : span_mask_tokens_dataset.py](https://github.com/facebookresearch/fairseq/blob/main/fairseq/data/span_mask_tokens_dataset.py)
