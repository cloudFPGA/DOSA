from brevitas.quant import Int8WeightPerTensorFixedPoint, Int8ActPerTensorFixedPoint


class Int5WeightPerTensorFixedPoint(Int8WeightPerTensorFixedPoint):
    bit_width = 5


class Int5ActPerTensorFixedPoint(Int8ActPerTensorFixedPoint):
    bit_width = 5


class Int4WeightPerTensorFixedPoint(Int8WeightPerTensorFixedPoint):
    bit_width = 4


class Int4ActPerTensorFixedPoint(Int8ActPerTensorFixedPoint):
    bit_width = 4


class Int3WeightPerTensorFixedPoint(Int8WeightPerTensorFixedPoint):
    bit_width = 3


class Int3ActPerTensorFixedPoint(Int8ActPerTensorFixedPoint):
    bit_width = 3
