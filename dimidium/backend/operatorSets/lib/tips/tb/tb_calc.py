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

inp = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
weights = np.array([[10, 11, 12, 13], [21, 22, 23, 24], [31, 32, 33, 34]])
bias = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

r1 = np.dot(inp, weights)
r2 = r1.flat + bias

print(r2)
# array([146, 152, 158, 164, 332, 347, 362, 377, 518, 542, 566, 590])

l1 = list(r2)[:8]
l1.reverse()
print(l1)
# [377, 362, 347, 332, 164, 158, 152, 146]
# [str(hex(i)) for i in l1]
# ['0x179', '0x16a', '0x15b', '0x14c', '0xa4', '0x9e', '0x98', '0x92']
l2 = list(r2)[8:]
l2.reverse()
print(l2)
# [590, 566, 542, 518]
# ['0x24e', '0x236', '0x21e', '0x206']
