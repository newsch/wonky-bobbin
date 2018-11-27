
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
% matplotlib inline
import io
import time
from numpy.lib.stride_tricks import as_strided
# load our pickled stream object containing the image data
import pickle
def getBayer(filename):
    with open(filename,'rb') as f:
        stream = pickle.load(f)
    # # alternatively, just open the jpeg file (also works)
    # with open('wall1.jpeg', 'rb') as f:
    #     stream = io.BytesIO(f.read())
    assert isinstance(stream, io.BytesIO)
    ver = 1  # we used a v1 camera module for this image. Use `2` for v2

    offset = {
        1: 6404096,
        2: 10270208,
        }[ver]
    data = stream.getvalue()[-offset:]
    assert data[:4] == b'BRCM'
    data = data[32768:]
    data = np.frombuffer(data, dtype=np.uint8)

    # For the V1 module, the data consists of 1952 rows of 3264 bytes of data.
    # The last 8 rows of data are unused (they only exist because the maximum
    # resolution of 1944 rows is rounded up to the nearest 16).
    #
    # For the V2 module, the data consists of 2480 rows of 4128 bytes of data.
    # There's actually 2464 rows of data, but the sensor's raw size is 2466
    # rows, rounded up to the nearest multiple of 16: 2480.
    #
    # Likewise, the last few bytes of each row are unused (why?). Here we
    # reshape the data and strip off the unused bytes.

    reshape, crop = {
        1: ((1952, 3264), (1944, 3240)),
        2: ((2480, 4128), (2464, 4100)),
        }[ver]
    data = data.reshape(reshape)[:crop[0], :crop[1]]

    # Horizontally, each row consists of 10-bit values. Every four bytes are
    # the high 8-bits of four values, and the 5th byte contains the packed low
    # 2-bits of the preceding four values. In other words, the bits of the
    # values A, B, C, D and arranged like so:
    #
    #  byte 1   byte 2   byte 3   byte 4   byte 5
    # AAAAAAAA BBBBBBBB CCCCCCCC DDDDDDDD AABBCCDD
    #
    # Here, we convert our data into a 16-bit array, shift all values left by
    # 2-bits and unpack the low-order bits from every 5th byte in each row,
    # then remove the columns containing the packed bits

    data = data.astype(np.uint16) << 2
    for byte in range(4):
        data[:, byte::5] |= ((data[:, 4::5] >> ((4 - byte) * 2)) & 0b11)
    data = np.delete(data, np.s_[4::5], 1)

    # Now to split the data up into its red, green, and blue components. The
    # Bayer pattern of the OV5647 sensor is BGGR. In other words the first
    # row contains alternating green/blue elements, the second row contains
    # alternating red/green elements, and so on as illustrated below:
    #
    # GBGBGBGBGBGBGB
    # RGRGRGRGRGRGRG
    # GBGBGBGBGBGBGB
    # RGRGRGRGRGRGRG
    #
    # Please note that if you use vflip or hflip to change the orientation
    # of the capture, you must flip the Bayer pattern accordingly

    rgb = np.zeros(data.shape + (3,), dtype=data.dtype)
    rgb[1::2, 0::2, 0] = data[1::2, 0::2] # Red
    rgb[0::2, 0::2, 1] = data[0::2, 0::2] # Green
    rgb[1::2, 1::2, 1] = data[1::2, 1::2] # Green
    rgb[0::2, 1::2, 2] = data[0::2, 1::2] # Blue

    uint16_to_uint8 = lambda a: (a * (255/1023)).astype(np.uint8)  # note, this only works b/c the values are actually 10-bit
    # uint16_to_uint8 = lambda a: (a >> 2).astype(np.uint8)  # or bit-shift as suggested at the end
    rgb8 = uint16_to_uint8(rgb)
    np.max(rgb8)
    return(rgb8)
