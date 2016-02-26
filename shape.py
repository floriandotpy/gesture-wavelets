import math
import glob
import re
import itertools
import numpy

from matplotlib import pyplot

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi

from skimage.filters import gabor_kernel
import cv2

def read_pgm(filename, byteorder='>'):
    """
    Return image data from a raw PGM file as numpy array.

    Format specification: http://netpbm.sourceforge.net/doc/pgm.html
    """
    with open(filename, 'rb') as f:
        buf = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buf).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    return numpy.frombuffer(buf,
                            dtype='u1' if int(maxval) < 256 else byteorder+'u2',
                            count=int(width)*int(height),
                            offset=len(header)
                            ).reshape((int(height), int(width)))


class Gabor(object):

    def convolve_raw(self, frame, frequency=0.5, theta=0):

        kernel = gabor_kernel(frequency, theta=theta)

        real = ndi.convolve(image, np.real(kernel), mode='wrap')
        imag = ndi.convolve(image, np.imag(kernel), mode='wrap')

        return real, imag

    def convolve(self, frame, frequency=0.4):

        # do not change original. not actually sure this is needed
        frame = np.array(frame, dtype=frame.dtype)

        theta = 0 # in (0, 1)
        kernel = gabor_kernel(frequency, theta=theta)
        return self.power(frame, kernel)

    def power(self, image, kernel):
        # Normalize images for better comparison.
        image = (image - image.mean()) / image.std()
        return np.sqrt(ndi.convolve(image, np.real(kernel), mode='wrap')**2 +
                       ndi.convolve(image, np.imag(kernel), mode='wrap')**2)


class Coords(object):


    def __init__(self, from_shape, to_rows=10, to_cols=10):
        width = from_shape[0]
        height = from_shape[1]

        self.row_distance = height / float(to_rows)
        self.col_distance = width / float(to_cols)

    def get(self, row, col):

        return float(row) * self.row_distance, float(col) * self.col_distance


class ShapeDescription(object):

    def __init__(self):

        self.gabor = Gabor()
        self.images = []

    def imshow(self, image, label=''):
        self.images.append((image, label))

    def show(self, columns=4):

        # init pyplot axes
        rows = int(math.ceil(len(self.images) / float(columns)))
        fig, axes = plt.subplots(nrows=rows, ncols=columns)
        axes = [item for sublist in axes for item in sublist] # axes is nested array. flatten it

        # plot images
        for (img, lbl), ax in zip(self.images, axes):
            ax.imshow(img, pyplot.cm.gray, interpolation='nearest')
            ax.set_title(lbl)

        # show finished plot
        pyplot.show()

    def find_contours(self, binary):
        copy = numpy.copy(binary)
        contours, _ = cv2.findContours(copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def largest_bounding_rect(self, contours):

        x, y, w, h = 0, 0, 0, 0

        for cnt in contours:
            rect = cv2.boundingRect(cnt)
            _1, _2, width, height = rect
            if width*height > w*h:
                x, y, w, h = rect

        return x, y, w, h

    def process(self, image, filename):

        # pyplot.figure()
        self.plotCount = 0

        # 1. Raw image
        self.imshow(image)

        # 2. Binary image (assuming bimodal greyvalue distribution)
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        self.imshow(binary)

        # 3. Contours

        # find contours
        contours = self.find_contours(binary)

        # draw contours
        image_contours = numpy.zeros_like(binary)
        cv2.drawContours(image_contours, contours, -1, (255, 255, 255), 1)

        # Bounding rect
        x, y, w, h = self.largest_bounding_rect(contours)
        cv2.rectangle(image_contours, (x, y), (x+w, y+h), (255, 255, 255), 2)
        self.imshow(image_contours)

        # Isolate hand / slice image
        image_hand = image[y:y+h, x:x+w]
        self.imshow(image_hand)

        rows = cols = 10
        coords = Coords(image_hand.shape, to_rows=rows, to_cols=cols)
        #print "COORDS (%d)" % len(coords), coords

        # Gabor filter
        frequencies = (0.7, 0.9, 1.0, 1.2, 1.3, 1.4, 2.0, 3.0, 4.0, 5.0, 10.0)
        thetas = (0,)
        results = np.zeros((rows+1, cols+1, len(frequencies), len(thetas), 2))
        for i, freq in enumerate(frequencies):

            for j, theta in enumerate(thetas):

                cp = numpy.copy(image_hand)
                real, imag = self.gabor.convolve_raw(cp, freq)

                for row in xrange(rows+1):
                    for col in xrange(cols+1):

                        row_, col_ = coords.get(row, col)

                        results[row][col][i][j][0] = real[row_][col_]
                        results[row][col][i][j][1] = imag[row_][col_]

                # cp = numpy.copy(image_hand)
                # convolved = self.gabor.convolve(cp, freq)
                # convolved = np.array(convolved, dtype=np.uint8)
                # convolved = cv2.equalizeHist(convolved)
                #
                # self.imshow(convolved, 'f = %2.2f' % freq)

        print results

        # self.show()


if __name__ == '__main__':

    files = glob.glob1('images', '*2.pgm')
    tool = ShapeDescription()

    print "%d files with dark background found" % len(files)

    for filename in files[:1]:

        filename = "images/%s" % filename
        image = read_pgm(filename, byteorder='<')
        tool.process(image, filename)
