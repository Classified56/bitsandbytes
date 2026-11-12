"""Paper-compatible LLM.int8 decomposition with optional real XDNA1 matmuls.

The pinned bitsandbytes operators perform quantization and dequantization.
The CPU correction is the same sparse-column gather/dequantize/addmm sequence
as bitsandbytes/backends/default/ops.py, without any cache probes.
"""
import numpy as np
import torch
from torch import nn


class NpuPool:
    """Persistent weights and reusable, shape-specific XRT input/output buffers."""
    def __init__(self, rows=128):
        import aie.iron as iron
        from aie.iron.device import from_name
        from aie.utils.hostruntime import set_current_device
        from ml_dtypes import bfloat16
        from single_core import single_core
        set_current_device(from_name('npu', n_cols=1))
        self.iron, self.bf16, self.design = iron, bfloat16, single_core
        self.rows = rows
        self.scratch = {}
        self.weights = []
        self.calls = 0

    def upload(self, weights, quantized):
        array = (weights.detach().contiguous().numpy() if quantized else
                 weights.detach().float().numpy().astype(self.bf16))
        # The kernel consumes a column-major logical B, stored physically [N,K].
        tensor = self.iron.tensor(array.reshape(-1), device='npu')
        self.weights.append(tensor)
        return tensor

    def mm(self, activation, weight, out_features, quantized):
        rows, k = activation.shape
        assert rows == self.rows and k % 32 == 0 and out_features % 32 == 0
        key = (rows, k, out_features, quantized)
        if key not in self.scratch:
            dtype_in = np.int8 if quantized else self.bf16
            dtype_out = np.int32 if quantized else np.float32
            a = self.iron.zeros(rows*k, dtype=dtype_in, device='npu')
            c = self.iron.zeros(rows*out_features, dtype=dtype_out, device='npu')
            kernel = self.design.specialize(
                M=rows, K=k, N=out_features, m=32, k=32, n=32,
                dtype_in_str='i8' if quantized else 'bf16',
                dtype_out_str='i32' if quantized else 'f32', b_col_maj=1)
            self.scratch[key] = (a, c, kernel)
        a, c, kernel = self.scratch[key]
        source = activation.detach().numpy()
        view = a.numpy_view().reshape(rows, k)
        view[:] = source if quantized else source.astype(self.bf16)
        kernel(a, weight, c)  # Synchronous real NPU dispatch; no CPU fallback.
        self.calls += 1
        result = c.numpy().reshape(rows, out_features).copy()
        return torch.from_numpy(result)

    def tensor_bytes(self):
        return (sum(w.nbytes for w in self.weights) +
                sum(a.nbytes+c.nbytes for a,c,_ in self.scratch.values()))


class PlainNpuLinear(nn.Module):
    def __init__(self, original, pool):
        super().__init__()
        self.out_features, self.in_features = original.weight.shape
        self.register_buffer('bias', None if original.bias is None else original.bias.detach().clone())
        self.pool = pool
        self.npu_weight = pool.upload(original.weight, False)

    def forward(self, x):
        shape = x.shape
        y = self.pool.mm(x.reshape(-1, shape[-1]), self.npu_weight, self.out_features, False)
        if self.bias is not None:
            y = y + self.bias
        return y.to(x.dtype).reshape(*shape[:-1], self.out_features)


class MixedInt8Linear(nn.Module):
    def __init__(self, original, pool=None, threshold=6.0):
        super().__init__()
        import bitsandbytes.functional as F
        self.out_features, self.in_features = original.weight.shape
        self.threshold, self.pool = threshold, pool
        cb, scb, _ = F.int8_vectorwise_quant(original.weight.detach().to(torch.float16))
        self.register_buffer('scb', scb)
        self.register_buffer('bias', None if original.bias is None else original.bias.detach().clone())
        self.npu_weight = None
        if pool is not None:
            self.npu_weight = pool.upload(cb, True)
            # CPU correction reads the same shared-DDR weight buffer. There is
            # no duplicate retained CPU weight tensor for hybrid inference.
            cb = torch.from_numpy(self.npu_weight.numpy_view().reshape(self.out_features, self.in_features))
        self.register_buffer('cb', cb)
        self.calls = 0
        self.outlier_columns = 0

    def forward(self, x):
        import bitsandbytes.functional as F
        shape = x.shape
        a = x.reshape(-1, shape[-1])
        ca, sca, cols = F.int8_vectorwise_quant(a.to(torch.float16), threshold=self.threshold)
        suba = subb = None
        if cols is not None and cols.numel():
            suba = a[:, cols].contiguous()
            subb = torch.ops.bitsandbytes.int8_vectorwise_dequant.default(
                self.cb[:, cols].contiguous(), self.scb).to(a.dtype).t()
            self.outlier_columns += cols.numel()
        if self.pool is None:
            product = torch.ops.bitsandbytes.int8_linear_matmul.default(ca, self.cb)
        else:
            product = self.pool.mm(ca, self.npu_weight, self.out_features, True)
        y = torch.ops.bitsandbytes.int8_mm_dequant.default(
            product, sca, self.scb, dtype=a.dtype, bias=self.bias)
        if subb is not None:
            y = y.addmm(suba, subb)
        self.calls += 1
        return y.reshape(*shape[:-1], self.out_features)


def replace_linears(model, dq, pool=None):
    selected = [(name, module) for name, module in model.named_modules()
                if isinstance(module, nn.Linear) and name.startswith('model.layers.')]
    assert len(selected) == 24*7, len(selected)
    for name, module in selected:
        parent_name, child = name.rsplit('.', 1)
        parent = model.get_submodule(parent_name)
        replacement = MixedInt8Linear(module, pool) if dq else PlainNpuLinear(module, pool)
        setattr(parent, child, replacement)
    return [name for name,_ in selected]
