import torch
from typing import Optional


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

    def reshuffle_stream(self,
                         packed_stream: torch.Tensor,
                         *,
                         generator: Optional[torch.Generator] = None,
                         seed: Optional[int] = None,
                         ) -> torch.Tensor:
        """
        随机重排 packed bit-stream 的比特位置，但保持每个元素的 1 的总数不变。
        形状输入: (..., seq_len//32) int64
        形状输出: (..., seq_len//32) int64
        """
        seq_len = self.seq_len
        assert packed_stream.dtype in (torch.int64, torch.long)
        assert packed_stream.shape[-1] == seq_len // 32, \
            f"expect last dim={seq_len // 32}, got {packed_stream.shape[-1]}"

        if seed is not None:
            if generator is None:
                generator = torch.Generator(device=packed_stream.device)
            generator.manual_seed(seed)

        # 1) 解包到比特: (..., num_words, 32)
        shifts = torch.arange(32, device=packed_stream.device, dtype=torch.int64)
        bits = ((packed_stream.unsqueeze(-1) >> shifts) & 1).to(torch.int64)  # 0/1
        num_words = seq_len // 32

        # 2) 展平到一条比特流并按最后一维随机置换: (..., seq_len)
        bits_flat = bits.reshape(*bits.shape[:-2], seq_len)

        # 为每个元素生成独立的随机 key，并据此打乱（均匀置换）
        # 注：rand_like 会沿用 dtype，故用 rand 指定 dtype=float
        keys = torch.rand(bits_flat.shape, device=bits_flat.device, generator=generator)
        perm = keys.argsort(dim=-1)  # 每个元素一套独立置换
        bits_flat_shuf = torch.gather(bits_flat, dim=-1, index=perm)

        # 3) 重新打包回 int64: (..., num_words, 32) -> (..., num_words)
        bits_shuf = bits_flat_shuf.reshape(*bits.shape[:-2], num_words, 32)
        weights = (1 << torch.arange(32, device=bits_shuf.device, dtype=torch.int64))
        packed_new = (bits_shuf * weights).sum(dim=-1)  # int64

        return packed_new

    def circular_shift(self, packed_stream: torch.Tensor) -> torch.Tensor:

        assert packed_stream.dtype in (torch.int64, torch.long)
        B, F, W = packed_stream.shape

        seq_len = self.seq_len
        device = packed_stream.device
        dtype_i64 = torch.int64

        f_idx = torch.arange(F, device=device, dtype=dtype_i64)
        s_bits = f_idx % seq_len
        q_words = (s_bits // 32) % W
        r_bits = s_bits % 32

        base = torch.arange(W, device=device, dtype=dtype_i64)  # [W]
        src_idx = (base.unsqueeze(0) - q_words.unsqueeze(1)) % W  # [F, W]
        src_idx = src_idx.expand(B, -1, -1)  # [B, F, W]
        cur = torch.gather(packed_stream, dim=-1, index=src_idx)  # [B, F, W]

        prev = torch.roll(cur, shifts=1, dims=-1)  # [B, F, W]

        r_bc = r_bits.view(1, F, 1).expand(B, F, W)  # [B, F, W]

        l_bc = torch.where(r_bc == 0, torch.tensor(32, device=device, dtype=dtype_i64), 32 - r_bc)

        part1 = torch.bitwise_right_shift(cur, r_bc)  # cur >> r
        part2 = torch.bitwise_left_shift(prev, l_bc)  # prev << (32 - r or 32)

        out = torch.bitwise_or(part1, part2)

        mask32 = (torch.tensor(1, dtype=dtype_i64, device=device) << 32) - 1
        out = torch.bitwise_and(out, mask32)

        return out

    def circular_shift_per_word(self, packed_stream: torch.Tensor) -> torch.Tensor:

        assert packed_stream.dtype in (torch.int64, torch.long)
        B, F, W = packed_stream.shape
        device = packed_stream.device
        dtype_i64 = torch.int64

        f_idx = torch.arange(F, device=device, dtype=dtype_i64)
        r_bits = f_idx % 32
        l_bits = (32 - r_bits) % 32

        r_bc = r_bits.view(1, F, 1).expand(B, F, W)
        l_bc = l_bits.view(1, F, 1).expand(B, F, W)

        right = torch.bitwise_right_shift(packed_stream, r_bc)
        left = torch.bitwise_left_shift(packed_stream, l_bc)
        out = torch.bitwise_or(right, left)

        mask32 = (torch.tensor(1, dtype=dtype_i64, device=device) << 32) - 1
        out = torch.bitwise_and(out, mask32)

        return out


if __name__ == "__main__":
    torch.set_printoptions(threshold=float('inf'))
    # torch.set_printoptions(linewidth=320)

    def unpack_bits(packed: torch.Tensor, seq_len: int) -> torch.Tensor:
        # restore the bitstream before packing
        assert seq_len % 32 == 0, "seq_len must be multiple of 32"
        shifts = torch.arange(32, device=packed.device, dtype=torch.int64)
        bits = (packed.unsqueeze(-1) >> shifts) & 1
        bits = bits.view(*packed.shape[:-1], seq_len)
        return bits.int()


    seq_len = 128
    trans = Transform(seq_len=seq_len)
    x = torch.rand(4, 10) * 2 - 1
    y = trans.f2s(x)


