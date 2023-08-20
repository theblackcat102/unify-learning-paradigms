import json
import io
import torch
import zstandard as zstd
from tqdm import tqdm
from transformers import AutoTokenizer
from text_denoising.collate_fn import DataCollatorForUL2
from torch.utils.data import IterableDataset, DataLoader

def chunks(lst, chunk_size):
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i+chunk_size]

class ZstDataset(IterableDataset):

    def __init__(self, file, tokenizer, max_length=512) -> None:
        super().__init__()
        self.file = [file] if isinstance(file, str) else file
        self.tokenizer = tokenizer
        self.max_length = max_length


    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            for filename in self.file:
                with open(filename, 'rb') as f:
                    cctx = zstd.ZstdDecompressor()
                    reader = io.BufferedReader(cctx.stream_reader(f))
                    for line in reader:
                        if line:
                            raw_text = json.loads(line)['text']
                            for chunk in chunks(raw_text, int(self.max_length*3.2)):
                                tokens = self.tokenizer(chunk)['input_ids']
                                if len(tokens) >= self.max_length:
                                    yield { 'input_ids': tokens[:self.max_length] }
        elif len(self.file) == 1:
            cnt = 0
            worker_id = worker_info.id
            total_workers = worker_info.num_workers
            with open(self.file[0], 'rb') as f:
                cctx = zstd.ZstdDecompressor()
                reader = io.BufferedReader(cctx.stream_reader(f))
                for line in reader:
                    if line:
                        raw_text = json.loads(line)['text']
                        for chunk in chunks(raw_text, int(self.max_length*3.2)):
                            tokens = self.tokenizer(chunk)['input_ids']
                            cnt += 1
                            if len(tokens) >= self.max_length and cnt % total_workers == worker_id:
                                yield { 'input_ids': tokens[:self.max_length] }
        else:
            worker_id = worker_info.id
            with open(self.file[worker_id], 'rb') as f:
                cctx = zstd.ZstdDecompressor()
                reader = io.BufferedReader(cctx.stream_reader(f))
                for line in reader:
                    if line:
                        raw_text = json.loads(line)['text']
                        for chunk in chunks(raw_text, int(self.max_length*3.2)):
                            tokens = self.tokenizer(chunk)['input_ids']
                            if len(tokens) >= self.max_length:
                                yield {'input_ids': tokens[:self.max_length]}

if __name__ == "__main__":
    import glob
    # download test.jsonl.zst from the-pile website

    tokenizer = AutoTokenizer.from_pretrained("bigscience/mt0-small")
    tokenizer.add_special_tokens({'bos_token': '<s>'})
    dataset = ZstDataset(list(glob.glob('/mnt/ssd/pythia/*.jsonl.zst')), tokenizer)
    # dataset = ZstDataset('/mnt/ssd/pythia_val/val.jsonl.zst', tokenizer, max_length=600)
    # mimic result from multiple dataset runs
    collate_fn = DataCollatorForUL2(tokenizer,
                                r_probability=0.5, r_denoising=True,
                                s_probability=0.5, s_denoising=False,
                                x_denoising=False, x_probability=0.0)
    dataloader = DataLoader(dataset, batch_size=256, collate_fn=collate_fn, num_workers=13)


    # benchmark iteration speed
    for batch in tqdm(dataloader):
        print(batch['input_ids'].shape)
        # print(batch["decoder_attention_mask"][0])
        # for (input, label) in zip(batch['decoder_input_ids'][:8], batch['labels'][:8]):
        #     print(tokenizer.decode(input[ input != 0]))
        #     print(tokenizer.decode(label[ label != -100]))
        #     print('----')

    # batch = [  ]
    # np_batch = collate_fn(batch, return_tensors='pt')
    # print(np_batch)

