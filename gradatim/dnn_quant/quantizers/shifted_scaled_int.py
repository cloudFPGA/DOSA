from brevitas.quant import ShiftedUint8ActPerTensorFloat


class ShiftedUint4ActPerTensorFloat(ShiftedUint8ActPerTensorFloat):
    bit_width = 4
