import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from itertools import combinations

from transform import Transform


class Slayer(ABC):
    def __init__(self, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.trans = Transform(seq_len)

    @staticmethod
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
        shifts = torch.arange(32, device=x.device, dtype=x.dtype)  # [0 … 31]
        bits = (x.unsqueeze(-1) >> shifts) & 1  # (batch_size, n, num_ints, 32)
        votes = bits.sum(dim=1)  # (batch_size, num_ints, 32)
        maj_bit = votes >= (n // 2 + 1)  # bool → majority mask

        # --------------- re-pack to integer ---------------
        weights = (1 << shifts).to(x.dtype)  # 2**shift
        packed = (maj_bit.to(x.dtype) * weights).sum(dim=-1)  # (batch_size, num_ints)

        return packed

    @staticmethod
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
            maj_results = Slayer.MAJ(grouped_flat)  # (batch_size * num_groups, num_ints)

            # Reshape back to batch format: (batch_size, num_groups, num_ints)
            current = maj_results.view(batch_size, num_groups, num_ints)

        # Return the final single bit-stream
        return current.squeeze(1)  # (batch_size, num_ints)

    @staticmethod
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

    @staticmethod
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
            maj_results = Slayer.MAJ_sim(reshaped)

            # reshape back
            x = maj_results.reshape(batch_size, num_chunks)

        return x

    @abstractmethod
    def Sforward(self, stream):
        pass



class SConv2d(Slayer, nn.Module):
    # make sure Slayer comes before nn.Module
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            bias=True,
            seq_len=1024,
            strict=True):

        '''
        comments for additional parameters:
        :param seq_len: length of bit-streams
        :param strict: whether to check the number of stackMAJ inputs is some power of maj_dim
        '''

        super().__init__(seq_len)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = nn.modules.utils._pair(kernel_size)
        self.stride = nn.modules.utils._pair(stride)
        self.padding = nn.modules.utils._pair(padding)
        self.dilation = nn.modules.utils._pair(dilation)

        kh, kw = self.kernel_size
        weight_shape = (out_channels, in_channels, kh, kw)
        self.weight = nn.Parameter(torch.empty(weight_shape))
        nn.init.kaiming_uniform_(self.weight)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)

        self.maj_dim = self.kernel_size[0]
        self.strict = strict

    def forward(self, x):
        N, C, H, W = x.shape
        kh, kw = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.padding
        dh, dw = self.dilation

        # output size
        H_out = (H + 2 * ph - dh * (kh - 1) - 1) // sh + 1
        W_out = (W + 2 * pw - dw * (kw - 1) - 1) // sw + 1

        # unfold the input tensor using F.unfold, getting tensor [N, C_in*kh*kw, L]
        # the "L" is the number of all possible sliding-window positions
        # the "C_in*kh*kw" is number of pixels of a certain [C_in, kh, kw] region of input tensor
        patches = F.unfold(x, kernel_size=self.kernel_size,
                           padding=self.padding, stride=self.stride)  # [N, C_in*kh*kw, L], where L = H_out * W_out
        weight_flat = self.weight.view(self.out_channels, -1)  # [C_out, C_in*kh*kw]

        # Here the purpose is to replace the usual summation with stackMAJ in 2D convolution,
        # so it's not appropriate to directly muptiply the two tensors. Instead, we do elementwise product first:
        prod = patches.unsqueeze(1) * weight_flat.unsqueeze(0).unsqueeze(-1)  # [N, C_out, C_in*kh*kw, L]
        #      [N, 1, C_in*kh*kw, L]   *       [1, C_out, C_in*kh*kw, 1]
        # a certain element prod[n, m, k, l] means the k-th elementwise product in the l-th sliding-window position in the m-th channel of the n-th sample

        # we shall next conduct stackMAJ to its 3rd dimension and squeeze this dimension
        out_unfolded = SConv2d.stackMAJ_conv(prod, self.maj_dim)
        out = out_unfolded.view(N, self.out_channels, H_out, W_out)
        return out


    def Sforward(self, stream):
        N, C, H, W, num_ints = stream.shape     # here, num_ints means seq_len//32
        kh, kw = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.padding
        dh, dw = self.dilation

        # output size
        H_out = (H + 2 * ph - dh * (kh - 1) - 1) // sh + 1
        W_out = (W + 2 * pw - dw * (kw - 1) - 1) // sw + 1

        # unfolded_manual = SConv2d.tensor_unfold_5d(stream, self.kernel_size, self.stride, self.padding)


        # To apply F.unfold to a 5D tensor, we need to first reshape to 4D and then reshape back to 5D
        # (1) reshape to [N*num_ints, C_in, H, W], and apply F.unfold
        stream = stream.to(torch.float64)   # F.unfold can only process float tensors
        stream_reshaped = stream.permute(0, 4, 1, 2, 3).contiguous().view(N * num_ints, C, H, W)
        unfolded = F.unfold(stream_reshaped, kernel_size=self.kernel_size, stride=self.stride,
                            padding=self.padding, dilation=self.dilation)   # [N*num_ints, C_in*kh*kw, L]
        # (2) reshape back
        _, feature_dim, L = unfolded.shape
        unfolded = unfolded.view(N, num_ints, feature_dim, L).permute(0, 2, 3, 1) # [N, C_in*kh*kw, L, num_ints]

        unfolded = unfolded.to(torch.int64)     # convert back to int64

        # prepare convolution kernel for stochastic forward propagation
        Sweight = self.trans.f2s(self.weight)   # [C_out, C_in, kh, kw, num_ints]
        Sweight = Sweight.view(self.out_channels, -1, num_ints) # [C_out, C_in*kh*kw, num_ints]

        # elementwise product using XNOR gates
        prod = torch.bitwise_xor(unfolded.unsqueeze(1), Sweight.unsqueeze(0).unsqueeze(-2))
        prod = torch.bitwise_not(prod)      # [N, C_out, C_in*kh*kw, L, num_ints]

        # conduct stackMAJ as summation
        prod = prod.permute(0, 1, 3, 2, 4)  # [N, C_out, L, C_in*kh*kw, num_ints]
        prod = prod.reshape(-1, prod.shape[-2], prod.shape[-1])     # [N*C_out*L, C_in*kh*kw, num_ints]
        out = Slayer.stackMAJ(prod, self.maj_dim, strict=self.strict)   # [N*C_out*L, num_ints]
        out = out.view(N, self.out_channels, L, num_ints)               # [N, C_out, L, num_ints]
        out = out.view(N, self.out_channels, H_out, W_out, num_ints)    # [N, C_out, H_out, W_out, num_ints]
        return out





if __name__ == '__main__':
    l = SConv2d(3, 3, 3, 1, seq_len = 64)
    x = torch.rand(4, 3, 224, 224)
    x = l.trans.f2s(x)
    # print(l.Sforward(x).shape)
    l.Sforward(x)