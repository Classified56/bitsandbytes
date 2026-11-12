"""Verify true NPU BF16/INT8 products and the exact paper DQ decomposition."""
import json
from pathlib import Path
import numpy as np
import torch
import bitsandbytes as bnb
import bitsandbytes.functional as F
from linear_backends import NpuPool, MixedInt8Linear

torch.set_num_threads(4)
torch.manual_seed(20261111)
out = Path(__file__).resolve().parents[1]/'results/kernel-validation.json'
pool = NpuPool(128)
records = []
with torch.inference_mode():
    for k,n in [(896,128), (4864,896)]:
        for quantized in (True, False):
            if quantized:
                x = torch.randint(-127,128,(128,k),dtype=torch.int8)
                w = torch.randint(-127,128,(n,k),dtype=torch.int8)
                expected = torch._int_mm(x, w.t())
            else:
                x = torch.randn(128,k).bfloat16().float()
                w = torch.randn(n,k).bfloat16().float()
                expected = x@w.t()
            bo = pool.upload(w, quantized)
            actual = pool.mm(x, bo, n, quantized)
            if quantized:
                assert torch.equal(actual,expected)
            else:
                torch.testing.assert_close(actual,expected,rtol=0.003,atol=0.003)
            record={'shape':[128,k,n],'dtype':'int8' if quantized else 'bf16',
                    'max_abs_error':float((actual.double()-expected.double()).abs().max())}
            print(json.dumps(record),flush=True)
            records.append(record)
    original=torch.nn.Linear(896,128).eval()
    cpu=MixedInt8Linear(original)
    npu=MixedInt8Linear(original,pool)
    x=torch.randn(128,896)
    x[0,17]=8.0
    x[127,821]=-9.0
    for case,values in [('outliers',x),('no_outliers',x.clamp(-1,1))]:
        ca,sca,cols=F.int8_vectorwise_quant(values.half(),threshold=6.0)
        reference,_=torch.ops.bitsandbytes.int8_mixed_scaled_mm.default(values,ca,cpu.cb,sca,cpu.scb,cols,cpu.bias)
        for backend,module in [('cpu',cpu),('npu',npu)]:
            actual=module(values)
            torch.testing.assert_close(actual,reference,rtol=1e-5,atol=1e-5)
            record={'case':case,'backend':backend,'outlier_columns':cols.numel(),
                    'max_abs_error':float((actual-reference).abs().max())}
            print(json.dumps(record),flush=True)
            records.append(record)
out.write_text(json.dumps({'passed':True,'records':records,'npu_dispatches':pool.calls},indent=2)+'\n')
