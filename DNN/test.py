import numpy as np
def identity(inp):
    return 2*inp
a = np.array([
    [
        [11,2],
        [3,9]
    ],
    [
        [5,12],
        [7,13]
    ]
])
c = np.argmax(a)
print(c)
# print(c.shape)

array = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
])#数组一定要通过np形成，不然会报错


b = array[0, :]
print(b)
print(b.shape)
#[[1 2 3 4]]