import torch


class Transform:
    """
    Transformer that transform float Tensor to Stream and the other way round
    """

    def __init__(self, seq_len):
        self.seq_len = seq_len

    def f2s(self, float_tensor: torch.Tensor) -> torch.Tensor:
        assert self.seq_len % 32 == 0, "seq_len should be multiple of 32"
        p = (float_tensor + 1) / 2
        dims = len(float_tensor.shape)
        p = p.unsqueeze(-1).expand(*(-1,) * dims, self.seq_len)
        bits = torch.bernoulli(p)
        bits = bits.view(*bits.shape[:-1], self.seq_len // 32, 32)
        weights = torch.tensor([1 << i for i in range(32)], dtype=torch.int64, device=bits.device)
        packed = (bits.int() * weights).sum(dim=-1)
        return packed


    def s2f(self, packed_stream: torch.Tensor) -> torch.Tensor:
        num_ints = packed_stream.shape[-1]
        assert (
                num_ints == self.seq_len // 32
        ), f"wrong number of ints, expect {self.seq_len // 32}, got {num_ints}"
        shifts = torch.arange(32, device=packed_stream.device, dtype=packed_stream.dtype)
        bits = (packed_stream.unsqueeze(-1) >> shifts) & 1
        popcount = bits.sum(dim=-1)
        total_ones = popcount.sum(dim=-1)
        p = total_ones / self.seq_len
        return p * 2 - 1




if __name__ == "__main__":
    trans = Transform(seq_len=64)
    x = torch.rand(4, 10)*2 - 1
    y = trans.f2s(x)
    z = trans.s2f(y)
