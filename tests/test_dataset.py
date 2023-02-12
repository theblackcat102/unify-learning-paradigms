import json
import io
import zstandard as zstd
from tqdm import tqdm
from transformers import AutoTokenizer
from collate_fn import DataCollatorForUL2
from torch.utils.data import IterableDataset, DataLoader

def chunks(lst, chunk_size):
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i+chunk_size]

class ZstDataset(IterableDataset):

    def __init__(self, file, tokenizer) -> None:
        super().__init__()
        self.file = file
        self.tokenizer = tokenizer


    def __iter__(self):
        with open(self.file, 'rb') as f:
            cctx = zstd.ZstdDecompressor()
            reader = io.BufferedReader(cctx.stream_reader(f))
            for line in reader:
                if line:
                    raw_text = json.loads(line)['text']
                    for chunk in chunks(self.tokenizer(raw_text)['input_ids'], 512):
                        yield {'input_ids': chunk}


if __name__ == "__main__":

    tokenizer = AutoTokenizer.from_pretrained("bigscience/mt0-small")
    tokenizer.bos_token_id = 0
    dataset = ZstDataset('test.jsonl.zst', tokenizer)
    # mimic result from multiple dataset runs
    collate_fn = DataCollatorForUL2(tokenizer)
    dataloader = DataLoader(dataset, batch_size=128, collate_fn=collate_fn)
    for batch in tqdm(dataloader):
        batch
    # batch = [  ]
    # np_batch = collate_fn(batch, return_tensors='pt')
    # print(np_batch)

    # t5_collate = DataCollatorForT5Pretraining(tokenizer)
    # np_batch = t5_collate(tokenizer(test))

    # for b in batch:
    #     print(b['input_ids'])
    #     if len(b['input_ids']) <= 3:
    #         continue
    #     print(b['input_ids'])
    #     noise_ = random_spans_noise_mask(len(b['input_ids']), 3, 0.15)
    #     print(noise_)