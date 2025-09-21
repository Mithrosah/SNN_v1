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
    def __init__(self):
        super().__init__()

    @staticmethod
    def MAJ3(x: torch.Tensor) -> torch.Tensor:
        """
        Majority gate for 3 packed bit-streams with batch support.

        Parameters
        ----------
        x : torch.Tensor, shape (batch_size, 3, seq_len//32)
            batch_size is any positive integer

        Returns
        -------
        torch.Tensor, shape (batch_size, seq_len//32), same dtype as x
            Packed majority bit-stream for each batch.
        """
        assert x.shape[1] == 3, "Second dimension must be 3"

        a, b, c = x[:, 0], x[:, 1], x[:, 2]
        return (a & b) | (a & c) | (b & c)

    @staticmethod
    def stackMAJ3(x: torch.Tensor, strict: bool = True) -> torch.Tensor:
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
                assert current_n % 3 == 0, f"Cannot divide {current_n} streams by 3 evenly"
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
            maj_results = Slayer.MAJ3(grouped_flat)  # (batch_size * num_groups, num_ints)

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
    def prepare_Sforward(self, trans):
        pass

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
            strict=True,
            polarize=False):

        '''
        comments for additional parameters:
        :param strict: whether to check the number of stackMAJ inputs is some power of maj_dim
        '''

        super().__init__()

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

        self.strict = strict
        self.polarize = polarize
        if polarize:
            self.kk = nn.Parameter(torch.ones(1), requires_grad=False)

    def set_kk(self, kknew):
        if self.polarize:
            with torch.no_grad():
                self.kk.data = torch.tensor([kknew]).to(self.kk.device)
        else:
            raise AttributeError('set_kk() can only be called when polarize is set to True')

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
        # note that here we split padding and unfolding apart in order to customize padding value
        padded = F.pad(x, pad=(pw, pw, ph, ph), mode='constant', value=-1)
        patches = F.unfold(padded, kernel_size=self.kernel_size,
                           padding=0, stride=self.stride)  # [N, C_in*kh*kw, L], where L = H_out * W_out

        if self.polarize:
            weight = torch.tanh(self.weight * self.kk)
        else:
            weight = self.weight

        weight_flat = weight.view(self.out_channels, -1)  # [C_out, C_in*kh*kw]

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

    def prepare_Sforward(self, trans):
        self.Sweight = trans.f2s(self.weight)   # [C_out, C_in, kh, kw, num_ints]

    def Sforward(self, stream):
        N, C, H, W, num_ints = stream.shape     # here, num_ints means seq_len//32
        kh, kw = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.padding
        dh, dw = self.dilation

        # output size
        H_out = (H + 2 * ph - dh * (kh - 1) - 1) // sh + 1
        W_out = (W + 2 * pw - dw * (kw - 1) - 1) // sw + 1


        # To apply F.unfold to a 5D tensor, we need to first reshape to 4D and then reshape back to 5D
        # (1) reshape to [N*num_ints, C_in, H, W], and apply F.pad & F.unfold
        stream = stream.to(torch.float64)   # F.unfold can only process float tensors
        stream_reshaped = stream.permute(0, 4, 1, 2, 3).contiguous().view(N * num_ints, C, H, W)
        stream_padded = F.pad(stream_reshaped, pad=(pw, pw, ph, ph), value=0)
        unfolded = F.unfold(stream_padded, kernel_size=self.kernel_size, stride=self.stride,
                            padding=0, dilation=self.dilation)   # [N*num_ints, C_in*kh*kw, L]

        # (2) reshape back
        _, feature_dim, L = unfolded.shape
        unfolded = unfolded.view(N, num_ints, feature_dim, L).permute(0, 2, 3, 1) # [N, C_in*kh*kw, L, num_ints]

        unfolded = unfolded.to(torch.int64)     # convert back to int64

        # prepare convolution kernel for stochastic forward propagation
        Sweight = self.Sweight.view(self.out_channels, -1, num_ints) # [C_out, C_in*kh*kw, num_ints]

        # elementwise product using XNOR gates
        prod = torch.bitwise_xor(unfolded.unsqueeze(1), Sweight.unsqueeze(0).unsqueeze(-2))
        prod = torch.bitwise_not(prod)      # [N, C_out, C_in*kh*kw, L, num_ints]

        # conduct stackMAJ as summation
        prod = prod.permute(0, 1, 3, 2, 4)  # [N, C_out, L, C_in*kh*kw, num_ints]
        prod = prod.reshape(-1, prod.shape[-2], prod.shape[-1])     # [N*C_out*L, C_in*kh*kw, num_ints]
        out = Slayer.stackMAJ3(prod, strict=self.strict)   # [N*C_out*L, num_ints]
        out = out.view(N, self.out_channels, L, num_ints)               # [N, C_out, L, num_ints]
        out = out.view(N, self.out_channels, H_out, W_out, num_ints)    # [N, C_out, H_out, W_out, num_ints]
        return out


class SAvgPool2d(Slayer, nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, strict=True):
        super().__init__()
        self.kernel_size = nn.modules.utils._pair(kernel_size)
        self.stride = nn.modules.utils._pair(stride or kernel_size)  # if stride is None, self.stride=kernel_size
        self.padding = nn.modules.utils._pair(padding)

        self.strict = strict

    def forward(self, x):
        N, C, H, W = x.shape
        kh, kw = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.padding

        # output size
        H_out = (H + 2 * ph - kh) // sh + 1
        W_out = (W + 2 * pw - kw) // sw + 1

        # unfold into patches
        padded = F.pad(x, pad=(pw, pw, ph, ph), mode='constant', value=-1)
        patches = F.unfold(padded, kernel_size=self.kernel_size,
                           padding=0, stride=self.stride)  # [N, C_in*kh*kw, L], where L = H_out * W_out

        # reshape
        patches = patches.view(N, C, kh * kw, -1)  # [N, C_in, kh*kw, L]
        patches = patches.permute(0, 1, 3, 2)  # [N, C_in, L, kh*kw]
        patches = patches.reshape(-1, patches.shape[-1])    # [N*C_in*L, kh*kw]

        # conduct stackMAJ
        unfolded = Slayer.stackMAJ3_sim(patches, strict=self.strict)    # [N*C_in*L, 1]

        # reshape back
        out = unfolded.reshape(N, C, H_out, W_out)  # [N, C, H_out, W_out]
        return out

    def prepare_Sforward(self, trans):
        pass

    def Sforward(self, stream):
        N, C, H, W, num_ints = stream.shape
        kh, kw = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.padding

        # output size
        H_out = (H + 2 * ph - kh) // sh + 1
        W_out = (W + 2 * pw - kw) // sw + 1

        # unfold
        stream = stream.to(torch.float64)   # F.unfold can only process float tensors
        stream_reshaped = stream.permute(0, 4, 1, 2, 3).contiguous().view(N * num_ints, C, H, W)
        stream_padded = F.pad(stream_reshaped, (pw, pw, ph, ph), mode='constant', value=0)
        unfolded = F.unfold(stream_padded, kernel_size=self.kernel_size, stride=self.stride,
                            padding=0)   # [N*num_ints, C_in*kh*kw, L]
        _, feature_dim, L = unfolded.shape
        unfolded = unfolded.view(N, num_ints, feature_dim, L).permute(0, 2, 3, 1) # [N, C_in*kh*kw, L, num_ints]
        unfolded = unfolded.to(torch.int64)  # convert back to int64

        # reshape and conduct stackMAJ
        unfolded = unfolded.reshape(N, C, kh*kw, L, num_ints).permute(0, 1, 3, 2, 4)    # [N, C_in, L, kh*kw, num_ints]
        unfolded = unfolded.reshape(N*C*L, kh*kw, num_ints) # [N*C_in*L, kh*kw, num_ints]
        out = Slayer.stackMAJ3(unfolded, strict=self.strict) # [N*C_in*L, num_ints]

        # reshape back
        out = out.reshape(N, C, L, num_ints)
        out = out.reshape(N, C, H_out, W_out, num_ints)
        return out



class SLinear(Slayer, nn.Module):
    def __init__(self, in_features, out_features, strict=True, summation=True, polarize=False):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        self.strict = strict
        self.summation = summation
        self.polarize = polarize
        if polarize:
            self.kk = nn.Parameter(torch.ones(1), requires_grad=False)

    def set_kk(self, kknew):
        if self.polarize:
            with torch.no_grad():
                self.kk.data = torch.tensor([kknew]).to(self.kk.device)
        else:
            raise AttributeError('set_kk() can only be called when polarize is set to True')

    def forward(self, x):
        '''
        :param x: [batch_size, in_features]
        :return: out: [batch_size, out_features] if summation else [batch_size, out_features, in_features]
        '''

        if self.polarize:
            weight = torch.tanh(self.weight * self.kk)
        else:
            weight = self.weight

        # elementwise product
        prod = x.unsqueeze(1) * weight.unsqueeze(0)  # [batch_size, out_features, in_features]

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

    def prepare_Sforward(self, trans):
        self.Sweight = trans.f2s(self.weight)  # [out_features, in_features, seq_len//32]

    def Sforward(self, stream):
        '''
        :param stream: [batch_size, in_features, seq_len//32]
        :return: out: [batch_size, out_features, seq_len//32] if summation else [batch_size, out_features, in_features, seq_len//32]
        '''

        batch_size, in_features, num_ints = stream.shape

        # elementwise product using xnor
        prod = torch.bitwise_xor(stream.unsqueeze(1), self.Sweight.unsqueeze(0))
        prod = torch.bitwise_not(prod)  # [batch_size, out_features, in_features, seq_len//32]

        # reshape
        prod = prod.reshape(batch_size * self.out_features, in_features,
                            num_ints)  # [batch_size*out_features, in_features, seq_len//32]

        # conduct stackMAJ
        out = Slayer.stackMAJ3(prod, strict=self.strict) # [batch_size*out_features, seq_len//32]
        out = out.reshape(batch_size, self.out_features, num_ints)  # [batch_size, self.out_features, num_ints]

        return out


class SActv(Slayer, nn.Module):
    def __init__(self, repeats):
        super().__init__()
        self.repeats = repeats      # should be some power of 3

    @staticmethod
    def right_shift_stack(packed: torch.Tensor, n: int, chunk_size: int = 16) -> torch.Tensor:
        '''
        theoretically, bigger chunk_size costs more GPU memory and is of course faster
        but not as measurable as expected.
        '''
        if n <= 0:
            raise ValueError("n must be greater than 0")

        original_shape = packed.shape
        num_ints = original_shape[-1]
        batch_dims = original_shape[:-1]
        seq_len = num_ints * 32

        unpack_shifts = torch.arange(31, -1, -1, device=packed.device, dtype=packed.dtype)
        bits = (packed.unsqueeze(-1) >> unpack_shifts) & 1
        bits = bits.view(*batch_dims, seq_len)

        base_indices = torch.arange(seq_len, device=packed.device)
        pack_weights = (1 << torch.arange(31, -1, -1, device=packed.device)).to(packed.dtype)

        output_chunks = []

        for i in range(0, n, chunk_size):
            start_shift = i
            end_shift = min(i + chunk_size, n)
            current_chunk_size = end_shift - start_shift

            shift_amounts = torch.arange(start_shift, end_shift, device=packed.device).unsqueeze(-1)

            rolled_indices = (base_indices - shift_amounts) % seq_len
            view_shape = (1,) * len(batch_dims) + (current_chunk_size, seq_len)
            expanded_indices = rolled_indices.view(view_shape).expand(*batch_dims, current_chunk_size, seq_len)
            source_for_gather = bits.unsqueeze(-2).expand_as(expanded_indices)
            shifted_bits_chunk = source_for_gather.gather(dim=-1, index=expanded_indices)

            reshaped_bits_chunk = shifted_bits_chunk.view(*batch_dims, current_chunk_size, num_ints, 32)
            packed_again_chunk = (reshaped_bits_chunk.long() * pack_weights).sum(dim=-1)

            output_chunks.append(packed_again_chunk.to(packed.dtype))

        return torch.cat(output_chunks, dim=-2)

    def forward(self, x):
        for _ in range(self.repeats):
            x = -0.5 * x ** 3 + 1.5 * x
        return x

    def prepare_Sforward(self, trans):
        pass

    def Sforward(self, stream):
        if self.repeats > 0:
            # shift and stack
            stack = self.right_shift_stack(stream, 3**self.repeats)  # [..., 3^n, seq_len//32]
            fronts = stack.shape[:-2]
            n, num_ints = stack.shape[-2], stack.shape[-1]

            # combine the front dimensions
            stack = stack.reshape(-1, n, num_ints)  # [X, 3^n, num_ints],
                                                    # where X is the product of the lengths of all other dimensions

            # conduct stackMAJ
            out = Slayer.stackMAJ3(stack, strict=True)  # [X, num_ints]

            # reshape back
            out = out.reshape(*fronts, num_ints)    # [..., num_ints]
        else:
            out = stream
        return out


if __name__ == '__main__':
    trans = Transform(512)

    l = SConv2d(3, 3, 3, 1, strict=True)
    x = torch.rand(4, 3, 28, 28)
    s = trans.f2s(x)

    print(l(x).shape)
    l.prepare_Sforward(trans)
    print(l.Sforward(s).shape)

    m = SLinear(27, 81, summation=False)
    z = torch.rand(16, 27)*2 - 1
    print(m(z).shape)

    avg = SAvgPool2d(3, stride=2, padding=0)
    avg.prepare_Sforward(trans)
    print(avg.Sforward(s).shape)

    actv = SActv(1)
    print(actv(x).shape)
    actv.prepare_Sforward(trans)
    print(actv.Sforward(s).shape)