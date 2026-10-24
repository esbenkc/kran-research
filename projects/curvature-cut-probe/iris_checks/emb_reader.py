"""Low-RSS reader for the Qwen2.5-1.5B-Instruct tied unembedding (bf16 in model.safetensors): plain file reads, no
mmap, so the process RSS stays near the chunk size (the room's <= 0.4 GB rule while the fine-tune runs)."""
import json, struct
from pathlib import Path
import numpy as np

SNAP = next(Path.home().glob('.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/*')) / 'model.safetensors'
V, DIM = 151936, 1536


class Emb:
    def __init__(self, path=SNAP):
        self.fh = open(path, 'rb')
        n = struct.unpack('<Q', self.fh.read(8))[0]
        self.hdr = json.loads(self.fh.read(n))
        self.base = 8 + n

    def _bf16(self, name, start_byte, count):
        info = self.hdr[name]
        assert info['dtype'] == 'BF16'
        self.fh.seek(self.base + info['data_offsets'][0] + start_byte)
        raw = np.frombuffer(self.fh.read(count * 2), dtype=np.uint16)
        return (raw.astype(np.uint32) << 16).view(np.float32)

    def rows(self, a, b):
        """unembedding rows a..b-1 as float64 (b - a, DIM)."""
        return self._bf16('model.embed_tokens.weight', a * DIM * 2, (b - a) * DIM).reshape(b - a, DIM).astype(np.float64)

    def gamma(self):
        return self._bf16('model.norm.weight', 0, DIM).astype(np.float64)

    def chunks(self, ch=1024):
        for a in range(0, V, ch):
            yield a, self.rows(a, min(V, a + ch))
