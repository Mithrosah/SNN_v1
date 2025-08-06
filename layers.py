import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
import math

from transform import Transform


class MAJ3Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)       # [1, 3]
            ctx.squeeze_back = True
        else:
            ctx.squeeze_back = False

        xs = (x + 1) * 0.5           # [B,3], a,b,c ∈ [0,1]
        ctx.save_for_backward(xs)

        a, b, c = xs[:, 0], xs[:, 1], xs[:, 2]
        r = a * b + a * c + b * c - 2 * a * b * c
        y = (2 * r - 1).unsqueeze(-1)   # [B,1]
        return y

    @staticmethod
    def backward(ctx, grad_output):
        (xs,) = ctx.saved_tensors
        a, b, c = xs[:, 0], xs[:, 1], xs[:, 2]

        g1 = b + c - 2 * b * c
        g2 = a + c - 2 * a * c
        g3 = a + b - 2 * a * b
        G = torch.stack([g1, g2, g3], dim=1)  # [B,3]

        if grad_output.dim() == 0:
            grad_output = grad_output.view(1, 1).expand_as(G)
        elif grad_output.dim() == 1:
            grad_output = grad_output.view(-1, 1)

        grad_input = G * grad_output  # [B,3]

        if ctx.squeeze_back:
            grad_input = grad_input.squeeze(0)

        return grad_input


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
    def stackMAJ(x: torch.Tensor, strict: bool = True) -> torch.Tensor:
        """
        Stacked majority-gate with batch support using tensorized operations.
        designed especially for 3-input majority gates

        Parameters
        ----------
        x : torch.Tensor, shape (batch_size, n, seq_len//32)
            batch_size is any positive integer
            n is the number of bit-streams; dtype must be the same signed/unsigned int32/64
            Must be reducible by 3 in each iteration if strict==True.
        strict : bool
            whether to check the number of stackMAJ inputs(x.size(1)) is some power of maj_dim

        Returns
        -------
        torch.Tensor, shape (batch_size, seq_len//32), same dtype as x
            Single packed majority bit-stream for each batch.
        """
        # --------------- sanity checks ---------------
        assert x.dim() == 3, "x must be 3-D (batch_size, n, seq_len//32)"

        batch_size, n, num_ints = x.shape
        current = x  # (batch_size, n, num_ints)

        # --------------- tensorized iterative majority reduction ---------------
        while current.shape[1] > 1:
            current_n = current.shape[1]

            if strict:
                # Check if current_n is divisible by maj_dim
                assert current_n % 3 == 0, "Cannot divide {current_n} streams by 3 evenly"
                num_groups = current_n // 3
            else:
                num_groups = current_n // 3
                # If we can't form even one group, keep the first stream
                if num_groups == 0:
                    current = current[:, :1, :]
                    break
                # If there's more than one group but with residuals, drop the residuals
                current = current[:, :num_groups * 3, :]

            # Reshape to group streams: (batch_size, num_groups, maj_dim, num_ints)
            grouped = current.view(batch_size, num_groups, 3, num_ints)

            # Apply majority gate to all groups simultaneously
            # Process each group: (batch_size * num_groups, maj_dim, num_ints)
            grouped_flat = grouped.view(batch_size * num_groups, 3, num_ints)

            # Apply majority_packed_batch to all groups at once
            maj_results = Slayer.MAJ(grouped_flat)  # (batch_size * num_groups, num_ints)

            # Reshape back to batch format: (batch_size, num_groups, num_ints)
            current = maj_results.view(batch_size, num_groups, num_ints)

        # Return the final single bit-stream
        return current.squeeze(1)  # (batch_size, num_ints)

    @staticmethod
    def stackMAJ3_sim(x: torch.Tensor, strict: bool = True) -> torch.Tensor:
        '''
        vectorized stackMAJ3, support batch processing
        :param x: [batch_size, n]
        :param maj_dim: integer, number of inputs of a single MAJ
        :return: result: [batch_size, 1]
        '''
        if x.dim() == 1:
            x = x.unsqueeze(0)

        batch_size = x.size(0)

        while x.size(1) > 1:
            current_length = x.size(1)

            if strict:
                assert current_length % 3 == 0, "number of stackMAJ inputs must be some power of maj_dim"
                num_chunks = current_length // 3
            else:
                num_chunks = current_length // 3
                if num_chunks == 0:
                    x = x[:, :1]
                    break
                x = x[:, :num_chunks * 3]

            # reshape to fit the input shape of MAJ
            x = x.reshape(batch_size, num_chunks, 3)
            x = x.reshape(batch_size * num_chunks, 3)

            # conduct MAJ
            maj_results = MAJ3Fn.apply(x)

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


        N, out_channels, _, L = prod.size()

        # move the target dimension(3rd) to the last, and reshape to fit the input shape of stackMAJ
        prod = prod.movedim(2, -1)  # (N, C_out, L, C_in*kh*kw)
        flat_tensor = prod.reshape(-1, prod.shape[-1])  # (N*C_out*L, C_in*kh*kw)

        # conduct stackMAJ
        maj_results = Slayer.stackMAJ3_sim(flat_tensor, strict=self.strict)     # [N*C_out*L, 1]

        # squeeze
        maj_results = maj_results.squeeze(1)        # [N*C_out*L, ]

        # again reshape back to [N, C_out, L]
        unfolded = maj_results.reshape(N, out_channels, L)

        out = unfolded.view(N, self.out_channels, H_out, W_out)     # [N, C_out, H_out, W_out]
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
        out = Slayer.stackMAJ(prod, strict=self.strict)   # [N*C_out*L, num_ints]
        out = out.view(N, self.out_channels, L, num_ints)               # [N, C_out, L, num_ints]
        out = out.view(N, self.out_channels, H_out, W_out, num_ints)    # [N, C_out, H_out, W_out, num_ints]
        return out


class SLinear(Slayer, nn.Module):
    def __init__(self, in_features, out_features, seq_len=1024, strict=True, summation=True):
        super().__init__(seq_len)

        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        '''
        :param x: [batch_size, in_features]
        :return: out: [batch_size, out_features] if summation else [batch_size, out_features, in_features]
        '''

        # elementwise product
        prod = x.unsqueeze(1) * self.weight.unsqueeze(0)  # [batch_size, out_features, in_features]

        if self.summation:

            # reshape
            N, out_features, in_features = prod.shape
            prod = prod.reshape(N*out_features, in_features)     # [batch_size*out_features, in_features]

            # conduct stackMAJ
            out = Slayer.stackMAJ3_sim(prod, strict=self.strict)   # [batch_size*out_features, 1]

            # reshape back
            out = out.reshape(N, out_features)      # [batch_size, out_features]

        else:
            out = prod

        return out

    def Sforward(self, stream):
        '''
        :param stream: [batch_size, in_features, seq_len//32]
        :return: out: [batch_size, out_features, seq_len//32] if summation else [batch_size, out_features, in_features, seq_len//32]
        '''

        batch_size, in_features, num_ints = stream.shape
        Sweight = self.trans.f2s(self.weight)       # [out_features, in_features, seq_len//32]

        # elementwise product using xnor
        prod = torch.bitwise_xor(stream.unsqueeze(1), Sweight.unsqueeze(0))
        prod = torch.bitwise_not(prod)  # [batch_size, out_features, in_features, seq_len//32]

        if self.summation:
            # reshape
            prod = prod.reshape(batch_size * self.out_features, in_features,
                                num_ints)  # [batch_size*out_features, in_features, seq_len//32]

            # conduct stackMAJ
            out = Slayer.stackMAJ(prod, strict=self.strict) # [batch_size*out_features, seq_len//32]
            out = out.reshape(batch_size, self.out_features, num_ints)  # [batch_size, self.out_features, num_ints]

        else:
            out = prod

        return out




if __name__ == '__main__':
    l = SConv2d(3, 3, 3, 1, seq_len = 64)
    x = torch.rand(3, 3, 224, 224)
    print(l(x).shape)



