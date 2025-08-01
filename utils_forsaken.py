import torch


## 以下三个函数的输入都是经过transform.f2s()处理的浮点数，形状为(seq_len//32,)
## 不支持批处理，只能处理单组数据。
## 项目初期编写，可用于简单测试。

# 3输入的MAJ
def _majority3_packed(a: torch.Tensor,
                     b: torch.Tensor,
                     c: torch.Tensor) -> torch.Tensor:

    assert a.shape == b.shape == c.shape
    assert a.dtype == b.dtype == c.dtype
    return (a & b) | (a & c) | (b & c)

# 5输入的MAJ
def _majority5_packed(a: torch.Tensor,
                     b: torch.Tensor,
                     c: torch.Tensor,
                     d: torch.Tensor,
                     e: torch.Tensor) -> torch.Tensor:


    assert a.shape == b.shape == c.shape == d.shape == e.shape, "shape mismatch"
    assert a.dtype == b.dtype == c.dtype == d.dtype == e.dtype, "dtype mismatch"

    return (
        (a & b & c) | (a & b & d) | (a & b & e) |
        (a & c & d) | (a & c & e) | (a & d & e) |
        (b & c & d) | (b & c & e) | (b & d & e) |
        (c & d & e)
    )

# 任意输入数量（但必须为奇数）的MAJ
def _majority_packed(x: torch.Tensor) -> torch.Tensor:
    """
    Majority-gate for an odd number n of packed bit-streams.

    Parameters
    ----------
    x : torch.Tensor, shape (n, seq_len//32)
        n is any odd integer; dtype must be the same signed/unsigned int32/64
        as produced by Transform.f2s.

    Returns
    -------
    torch.Tensor, shape (seq_len//32,), same dtype as x
        Packed majority bit-stream.
    """
    # --------------- sanity checks ---------------
    assert x.dim() == 2,  "x must be 2-D (n, seq_len//32)"
    n, num_ints = x.shape
    assert n % 2 == 1,    "n must be odd so majority is well-defined"

    # --------------- bitwise majority ---------------
    shifts  = torch.arange(32, device=x.device, dtype=x.dtype)          # [0 … 31]
    bits    = (x.unsqueeze(-1) >> shifts) & 1                           # (n, num_ints, 32)
    votes   = bits.sum(dim=0)                                           # (num_ints, 32)
    maj_bit = votes >= (n // 2 + 1)                                     # bool → majority mask

    # --------------- re-pack to integer ---------------
    weights = (1 << shifts).to(x.dtype)                                 # 2**shift
    packed  = (maj_bit.to(x.dtype) * weights).sum(dim=-1)               # (num_ints,)

    return packed
