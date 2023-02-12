from transformers import AutoTokenizer
from utils import random_spans_noise_mask
from collate_fn import DataCollatorForUL2
with open('README.md', 'r') as f:
    test = f.readlines()


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("bigscience/mt0-small")
    tokenizer.bos_token_id = 0
    # mimic result from multiple dataset runs
    batch = [ tokenizer(t, ) for t in test if len(tokenizer(t,)['input_ids']) > 3 ]
    collate_fn = DataCollatorForUL2(tokenizer)
    np_batch = collate_fn(batch, return_tensors='np')
    print(np_batch)
    # t5_collate = DataCollatorForT5Pretraining(tokenizer)
    # np_batch = t5_collate(tokenizer(test))

    # for b in batch:
    #     print(b['input_ids'])
    #     if len(b['input_ids']) <= 3:
    #         continue
    #     print(b['input_ids'])
    #     noise_ = random_spans_noise_mask(len(b['input_ids']), 3, 0.15)
    #     print(noise_)