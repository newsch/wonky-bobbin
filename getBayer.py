import io
import time

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from scipy.ndimage import convolve


# numpy representation of the RPi camera module v1 Bayer Filter
bayerGrid = np.zeros((1944, 2592, 3), dtype=np.uint8)
bayerGrid[1::2, 0::2, 0] = 1 # Red
bayerGrid[0::2, 0::2, 1] = 1 # Green
bayerGrid[1::2, 1::2, 1] = 1 # Green
bayerGrid[0::2, 1::2, 2] = 1 # Blue

def get_bayer_grid(width: int, height: int):
    """Create a Numpy bayer grid representation for a given size.

    Uses the BGGR pattern of the Raspberry Pi camera sensor.
    """
    bayerGrid = np.zeros((height, width, 3), dtype=bool)
    bayerGrid[1::2, 0::2, 0] = 1 # Red
    bayerGrid[0::2, 0::2, 1] = 1 # Green
    bayerGrid[1::2, 1::2, 1] = 1 # Green
    bayerGrid[0::2, 1::2, 2] = 1 # Blue
    return bayerGrid


def debayerize(bayer_data, bayer_grid=None):
    """Interpolate bayer data of an image using nearest-neighbor."""
    if bayer_grid is None:
        h, w, d = bayer_data.shape
        bayer_grid = get_bayer_grid(width=w, height=h)
    kernel = np.ones((3,3), dtype=np.float)
    data_conv = rgb_convolve(bayer_data, kernel)
    grid_conv = rgb_convolve(bayer_grid, kernel)
    interpolated = data_conv / grid_conv
    # fill in missing data in bayer_data with interpolated
    result = bayer_data.copy()
    result[bayer_grid == 0] = interpolated[bayer_grid == 0]
    return result


def get_rgb_array(fp, dtype=np.uint64, width: int = None, height: int = None):
    """Return a 3-dimensional RGB numpy array of an image."""
    im = Image.open(fp)
    if height is not None or width is not None:
        cwidth, cheight = im.size
        if width is None:
            width = int(height * cwidth/cheight)
        elif height is None:
            height = int(width * cheight/cwidth)
        im = im.resize((width, height))
    return np.array(im, dtype=dtype)

def get_bw_array(fp, dtype=np.uint64, width: int = None, height: int = None):
    """Return a 2-dimensional black-and-white numpy array of an image."""
    a = get_rgb_array(fp, dtype=dtype, width=width, height=height)
    return np.mean(a, axis=2)

def rgb_convolve(image, kernel, mode='constant', cval=0.0, **kwargs):
    """Apply a convolution kernel to the RGB layers of an image independently.

    This applies scipy.ndimage.convolve with any additional parameters to the R,
    G, and B slices of array `image`.

    :param image: 3-dimensional numpy array of the image.
    :param kernel: 2-dimensional numpy array to convolve with the image.
    """
    res = np.zeros(image.shape, dtype=image.dtype)
    for i in range(3):
        res[:,:,i] = convolve(image[:,:,i], kernel, mode=mode, cval=cval, **kwargs)
    return res


def getBayer(filename: str, ver: int = 1):
    """Return the Bayer data from an RPi camera image.

    Note: this requires the Bayer output to be appended to the end of the
    image file. This can be done from the commandline by passing the `--raw`
    flag into the `raspistill` program and from the `picamera` Python module by
    passing `bayer=True` into the `PiCamera.capture` function.

    This uses code from the `picamera` module's documentation section on "Raw
    Bayer data captures". See https://picamera.readthedocs.io/en/release-1.13/recipes2.html#raw-bayer-data-captures

    :param ver: Version of the Raspberry Pi camera. Either 1 or 2.
    """
    offset = {
        1: 6404096,
        2: 10270208,
        }[ver]

    # open file and extract bayer data
    with open(filename, 'rb') as f:
        data = f.read()[-offset:]

    assert data[:4] == b'BRCM', "Could not find bayer data header"
    data = data[32768:]  # strip header data
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

    return rgb8


def debayer_malvar(img):
    """Debayer an image using the method proposed in Malvar et al.

    This method uses different filters for different cells in the
    The default values for alpha, beta, and gamma are the approximate Wiener
    values proposed by the paper.
    """
    def norm(k):
        return k / np.sum(k)

    # kernels for processing
    GatR = np.array([[0,0,-1,0,0],
                    [0,0,2,0,0],
                    [-1,2,4,2,-1],
                    [0,0,2,0,0],
                    [0,0,-1,0,0]])  # Green at red pixels
    GatB = GatR
    RatGRB = np.array([[0,0,.5,0,0],
                    [0,-1,0,-1,0],
                    [-1,4,5,4,-1],
                    [0,-1,0,-1,0],
                    [0,0,.5,0,0]])  # Red at Green, in Red row, Blue column
    RatGBR = RatGRB.T
    BatGBR = RatGRB
    BatGRB = RatGBR
    RatB = np.array([[0,0,-1.5,0,0],
                    [0,2,0,2,0],
                    [-1.5,0,6,0,-1.5],
                    [0,2,0,2,0],
                    [0,0,-1.5,0,0]])
    BatR = RatB

    # slices for grabbing specific colors
    Grows1 = slice(None,None,2)
    Gcols1 = Grows1
    Grows2 = slice(1,None,2)
    Gcols2 = Grows2

    Rrows = slice(1,None,2)
    Rcols = slice(None,None,2)

    Brows = slice(None,None,2)
    Bcols = slice(1,None,2)

    # indices for colors (cols, rows)
    # ideally these could be used directly, but the star operator doesn't seem
    # to work with slicing
    # iGatR = (Rcols,Rrows)
    # iGatB = (Bcols,Brows)
    # iRatGRB = (Gcols1,Grows1)
    # iBatGRB = iRatGRB
    # iRatGBR = (Gcols2,Grows2)
    # iBatGBR = iRatGBR
    # iRatB = (Bcols,Brows)
    # iBatR = (Rcols,Rrows)

    # actual demosaicing
    b = img.copy().sum(axis=2)  # flatten bayer data into 2-dimensional array
    debayered = img.copy()  # array to return, initially bayer data

    dGatR = convolve(b, norm(GatR))  # data for GatR and GatB
    debayered[Rrows,Rcols,1] = dGatR[Rrows,Rcols]
    debayered[Brows,Bcols,1] = dGatR[Brows,Bcols]

    dRatB = convolve(b, norm(RatB))  # data for RatB and BatR
    debayered[Brows,Bcols,0] = dRatB[Brows,Bcols]
    debayered[Rrows,Rcols,2] = dRatB[Rrows,Rcols]

    dRatGRB = convolve(b, norm(RatGRB))  # data for RatGRB and BatGBR
    debayered[Grows1,Gcols1,0] = dRatGRB[Grows1,Gcols1]
    debayered[Grows2,Gcols2,2] = dRatGRB[Grows2,Gcols2]

    dRatGBR = convolve(b, norm(RatGBR))  # data for RatGBR and BatGRB
    debayered[Grows2,Gcols2,0] = dRatGBR[Grows2,Gcols2]
    debayered[Grows1,Gcols1,2] = dRatGBR[Grows1,Gcols1]

    return debayered