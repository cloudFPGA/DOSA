import brevitas.nn as qnn
import torch
from brevitas.quant import Int8ActPerTensorFloat, ShiftedUint8ActPerTensorFloat, Int8ActPerTensorFloatMinMaxInit

from src.quantizers import Int8ActPerTensorFloatMax

act_quant = Int8ActPerTensorFloatMax
qidd = qnn.QuantIdentity(act_quant=act_quant, return_quant_tensor=True)

input0 = torch.zeros(2, 3)

input1 = torch.zeros(2, 3)
input1.data[1, 1] = 0.5

input2 = torch.full((2, 3), 1.0)

input3 = torch.arange(-3.0, 3.0, 1.0).reshape(-1, 3)

input4 = torch.arange(-0.5, 11.5, 2.0).reshape(-1, 3)

# === test ===
res0 = qidd(input0)
res1 = qidd(input1)
res2 = qidd(input2)
print('0: scale={}, zero-point={}'.format(res0.scale, res0.zero_point))
print('1: scale={}, zero-point={}'.format(res1.scale, res1.zero_point))
print('2: scale={}, zero-point={}'.format(res2.scale, res2.zero_point))
print('-- eval --')
qidd.eval()
res0 = qidd(input0)
res1 = qidd(input1)
res2 = qidd(input2)
print('0: scale={}, zero-point={}'.format(res0.scale, res0.zero_point))
print('1: scale={}, zero-point={}'.format(res1.scale, res1.zero_point))
print('2: scale={}, zero-point={}'.format(res2.scale, res2.zero_point))

print('\nreset..\n')
qidd = qnn.QuantIdentity(act_quant=act_quant, return_quant_tensor=True)
res2 = qidd(input2)
res3 = qidd(input3)
print('2: scale={}, zero-point={}'.format(res2.scale, res2.zero_point))
print('3: scale={}, zero-point={}'.format(res3.scale, res3.zero_point))
print('-- eval --')
qidd.eval()
res0 = qidd(input0)
res1 = qidd(input1)
res2 = qidd(input2)
print('0: scale={}, zero-point={}'.format(res0.scale, res0.zero_point))
print('1: scale={}, zero-point={}'.format(res1.scale, res1.zero_point))
print('2: scale={}, zero-point={}'.format(res2.scale, res2.zero_point))


print()
