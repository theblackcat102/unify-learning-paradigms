import json
import io
from tqdm import tqdm
import zstandard as zstd


if __name__ == "__main__":

    count = 0
    with open('/mnt/ssd/pythia/04.jsonl.zst', 'rb') as f:
        cctx = zstd.ZstdDecompressor()
        reader = io.BufferedReader(cctx.stream_reader(f))
        for line in tqdm(reader):
            count += 1
    print(count)

