import torch
import torch.nn.functional as F
from itertools import combinations

# 通用MAJ，任意奇数个输入比特流
def MAJ(x: torch.Tensor) -> torch.Tensor:
    """
    Majority-gate for an odd number n of packed bit-streams with batch support.

    Parameters
    ----------
    x : torch.Tensor, shape (batch_size, n, seq_len//32)
        batch_size is any positive integer
        n is any odd integer; dtype must be the same signed/unsigned int32/64
        as produced by Transform.f2s.

    Returns
    -------
    torch.Tensor, shape (batch_size, seq_len//32), same dtype as x
        Packed majority bit-stream for each batch.
    """
    # --------------- sanity checks ---------------
    assert x.dim() == 3, "x must be 3-D (batch_size, n, seq_len//32)"
    batch_size, n, num_ints = x.shape
    assert n % 2 == 1, "n must be odd so majority is well-defined"

    # --------------- bitwise majority ---------------
    shifts = torch.arange(32, device=x.device, dtype=x.dtype)              # [0 … 31]
    bits = (x.unsqueeze(-1) >> shifts) & 1                                 # (batch_size, n, num_ints, 32)
    votes = bits.sum(dim=1)                                                 # (batch_size, num_ints, 32)
    maj_bit = votes >= (n // 2 + 1)                                         # bool → majority mask

    # --------------- re-pack to integer ---------------
    weights = (1 << shifts).to(x.dtype)                                     # 2**shift
    packed = (maj_bit.to(x.dtype) * weights).sum(dim=-1)                    # (batch_size, num_ints)

    return packed

# 通用stackMAJ，使用了上面的通用MAJ
def stackMAJ(x: torch.Tensor, maj_dim: int, strict: bool = True) -> torch.Tensor:
    """
    Stacked majority-gate with batch support using tensorized operations.
    Requires the number of bit-streams to be reducible by maj_dim without remainder.

    Parameters
    ----------
    x : torch.Tensor, shape (batch_size, n, seq_len//32)
        batch_size is any positive integer
        n is the number of bit-streams; dtype must be the same signed/unsigned int32/64
        Must be reducible by maj_dim in each iteration.
    maj_dim : int
        Number of bit-streams to group for each majority operation. Must be odd.

    Returns
    -------
    torch.Tensor, shape (batch_size, seq_len//32), same dtype as x
        Single packed majority bit-stream for each batch.
    """
    # --------------- sanity checks ---------------
    assert x.dim() == 3, "x must be 3-D (batch_size, n, seq_len//32)"
    assert maj_dim % 2 == 1, "maj_dim must be odd so majority is well-defined"
    assert maj_dim >= 3, "maj_dim must be at least 3"

    batch_size, n, num_ints = x.shape
    current = x  # (batch_size, n, num_ints)

    # --------------- tensorized iterative majority reduction ---------------
    while current.shape[1] > 1:
        current_n = current.shape[1]

        if strict:
            # Check if current_n is divisible by maj_dim
            assert current_n % maj_dim == 0, f"Cannot divide {current_n} streams by {maj_dim} evenly"
            num_groups = current_n // maj_dim
        else:
            num_groups = current_n // maj_dim
            # If we can't form even one group, keep the first stream
            if num_groups == 0:
                current = current[:, :1, :]
                break
            # If there's more than one group but with residuals, drop the residuals
            current = current[:, :num_groups * maj_dim, :]

        # Reshape to group streams: (batch_size, num_groups, maj_dim, num_ints)
        grouped = current.view(batch_size, num_groups, maj_dim, num_ints)

        # Apply majority gate to all groups simultaneously
        # Process each group: (batch_size * num_groups, maj_dim, num_ints)
        grouped_flat = grouped.view(batch_size * num_groups, maj_dim, num_ints)

        # Apply majority_packed_batch to all groups at once
        maj_results = MAJ(grouped_flat)  # (batch_size * num_groups, num_ints)

        # Reshape back to batch format: (batch_size, num_groups, num_ints)
        current = maj_results.view(batch_size, num_groups, num_ints)

    # Return the final single bit-stream
    return current.squeeze(1)  # (batch_size, num_ints)

# 通用模拟MAJ，任意奇数个输入
def MAJ_sim(x: torch.Tensor) -> torch.Tensor:
    '''
    vectorized MAJ, support batch processing
    :param x: [batch_size, n]
    :return: result: [batch_size,]
    '''
    if x.dim() == 1:
        x = x.unsqueeze(0)

    batch_size, n = x.shape
    majority_threshold = (n + 1) // 2

    # pre-compute 1 + x and 1 - x
    one_plus_x = 1 + x  # shape: (batch_size, n)
    one_minus_x = 1 - x  # shape: (batch_size, n)

    result = torch.zeros(batch_size, dtype=x.dtype, device=x.device)

    # compute for every possible majority number in [majority_threshold, n]
    for k in range(majority_threshold, n + 1):
        # compute for every possible combinations that sums up to k
        for combo in combinations(range(n), k):
            combo_mask = torch.zeros(n, dtype=torch.bool, device=x.device)
            combo_mask[list(combo)] = True

            term = torch.ones(batch_size, dtype=x.dtype, device=x.device)

            # vectorized product
            term *= torch.prod(torch.where(combo_mask, one_plus_x, one_minus_x), dim=1)
            result += term

    return result / (2 ** (n - 1)) - 1

# 通用模拟stackMAJ，使用了上面的通用模拟MAJ
def stackMAJ_sim(x: torch.Tensor, maj_dim: int) -> torch.Tensor:
    '''
    vectorized stackMAJ, support batch processing
    :param x: [batch_size, n]
    :param maj_dim: integer, number of inputs of a single MAJ
    :return: result: [batch_size,]
    '''

    if x.dim() == 1:
        x = x.unsqueeze(0)

    batch_size = x.size(0)

    while x.size(1) >= maj_dim:
        current_length = x.size(1)
        num_chunks = current_length // maj_dim

        if num_chunks == 0:
            break

        # reshape to fit the input shape of MAJ
        reshaped = x[:, :num_chunks * maj_dim].reshape(batch_size * num_chunks, maj_dim)

        # conduct MAJ
        maj_results = MAJ_sim(reshaped)

        # reshape back
        x = maj_results.reshape(batch_size, num_chunks)

    return x

# 对比特流形式的图像数据[N, C, H, W, num_ints]做F.unfold
def unfold_5d(stream, kernel_size, stride, padding):
    """
    F.unfold() can only process 4D float tensors
    Here we mannualy implement F.unfold so that we can handle 5D int64 tensor
    **currently no support for dilation as a parameter**

    :param stream: [N, C, H, W, seq_len//32]
    :param kernel_size: kernel size
    :param stride: stride
    :param padding: padding
    :return: unfolded: [N, C*kh*kw, L, seq_len//32], where L = H_out * W_out
    """
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)

    N, C, H, W, num_ints = stream.shape
    kh, kw = kernel_size
    sh, sw = stride
    ph, pw = padding

    # add padding
    if padding != (0, 0):
        stream = F.pad(stream, (0, 0, pw, pw, ph, ph), mode='constant', value=0)
        _, _, H, W, _ = stream.shape

    # unfold tensor
    stream = stream.unfold(2, kh, sh).unfold(3, kw, sw)   # [N, C, out_H, out_W, num_ints, kH, kW]

    N, C, out_H, out_W, num_ints, kh, kw = stream.shape

    # reshape
    stream = stream.permute(0, 1, 5, 6, 2, 3, 4).contiguous()  # [N, C, kH, kW, out_H, out_W, num_ints]
    stream = stream.view(N, C * kh * kw, out_H * out_W, num_ints)  # [N, C*kH*kW, L, num_ints]

    return stream

# 三输入MAJ，输入参数是三个比特流堆叠形成的
def maj3_packed(inputs: torch.Tensor) -> torch.Tensor:
    assert inputs.shape[1] == 3, "Second dimension must be 3"

    a, b, c = inputs[:, 0], inputs[:, 1], inputs[:, 2]
    return (a & b) | (a & c) | (b & c)

# 比特流右移（向低位移动，看上去是左移）
def right_shift(packed: torch.Tensor) -> torch.Tensor:
    mask32 = torch.tensor((1 << 32) - 1, dtype=packed.dtype, device=packed.device)
    unsigned = packed & mask32  # (..., N)
    half = unsigned >> 1  # (..., N)
    carry = (torch.roll(unsigned, shifts=1, dims=-1) & 1) << 31  # (..., N)
    return (half | carry).to(packed.dtype)

# 比特流依次右移n次并堆叠成新的张量，使用了for循环和上面的right_shift函数
def right_shift_stack(packed, n):
    '''
    shift the input bit-stream rightwards by 0, 1, 2, ..., n bit circularly
    and stack the these results as the penultimate dimension.
    Note that "rightwards" here means toward the direction of higher bit positions(say, from 2^0 to 2^1)
    :param packed: bit stream that needs to be shifted and stacked. length of dim -1 must be seq_len//32
    :param n: times of shifts
    :return: [... , n, seq_len//32]
    '''
    if n <= 0:
        raise ValueError("n must be > 0")

    outs = [packed]
    cur = packed
    for _k in range(1, n):
        cur = right_shift(cur)
        outs.append(cur)

    return torch.stack(outs, dim=-2)
