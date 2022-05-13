#  *
#  *                       cloudFPGA
#  *     Copyright IBM Research, All Rights Reserved
#  *    =============================================
#  *     Created: Apr 2022
#  *     Authors: NGL
#  *
#  *     Description:
#  *
#  *

import numpy as np

# inp = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# weights = np.array([[10, 11, 12, 13], [21, 22, 23, 24], [31, 32, 33, 34]])
# bias = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
#
# r1 = np.dot(inp, weights)
# # array([[145, 151, 157, 163],
# #        [331, 346, 361, 376],
# #        [517, 541, 565, 589]])
# r2 = r1.flat + bias
#
# print(r2)
# # array([146, 152, 158, 164, 332, 347, 362, 377, 518, 542, 566, 590])
#
# l1 = list(r2)[:8]
# l1.reverse()
# print(l1)
# # [377, 362, 347, 332, 164, 158, 152, 146]
# # [str(hex(i)) for i in l1]
# # ['0x179', '0x16a', '0x15b', '0x14c', '0xa4', '0x9e', '0x98', '0x92']
# l2 = list(r2)[8:]
# l2.reverse()
# print(l2)
# # [590, 566, 542, 518]
# # ['0x24e', '0x236', '0x21e', '0x206']

inp = np.array([1, 2, 3, 4])
weights = np.array([[10, 11, 12, 13], [21, 22, 23, 24], [31, 32, 33, 34]])
bias = np.array([1, 1, 1])

r1 = np.dot(inp, weights.T)
# array([120, 230, 330])
r2 = r1 + bias
# array([121, 231, 331])

l1 = list(r2)
l1.reverse()
print([str(hex(i)) for i in l1])
# ['0x14b', '0xe7', '0x79']
# fixed point with 2 fractional bits
print([str(hex(i*4)) for i in l1])
# ['0x52c', '0x39c', '0x1e4']

li = list(inp)
li.reverse()
print([str(hex(i*4)) for i in li])
# ['0x10', '0xc', '0x8', '0x4']

wl = list(weights.flat)
print([str(hex(i*4)) for i in wl])
# ['0x28', '0x2c', '0x30', '0x34', '0x54', '0x58', '0x5c', '0x60', '0x7c', '0x80', '0x84', '0x88']

