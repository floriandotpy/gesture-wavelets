import math
import glob
import re

import numpy
import cv2
import pywt

from matplotlib import pyplot
from scipy import signal


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


class HandProcessor(object):

    def __init__(self, showwindow=True, save=False):

        self.showwindow = showwindow
        self.save = save

        self.filenumber = 0
        self.vectors = []

    def process(self, image, filename):

        print "Processing..."

        pyplot.figure()
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
        self.imshow(image_contours)

        # 4. Centroid
        image_moments = numpy.copy(image_contours)
        moments = cv2.moments(image_moments)
        centroid = (int(moments['m10']/moments['m00']), int(moments['m01']/moments['m00']))
        cv2.circle(image_moments, centroid, 3, (255, 255, 255), -1)

        # Bounding rect
        x, y, w, h = self.largest_bounding_rect(contours)
        cv2.rectangle(image_moments, (x, y), (x+w, y+h), (255, 255, 255), 2)
        self.imshow(image_moments)

        # Isolate hand / slice image
        image_hand = binary[y:y+h, x:x+w]
        centroid_hand = (centroid[0]-x, centroid[1]-y)
        cv2.circle(image_hand, centroid_hand, 3, (0, 0, 0), -1)
        self.imshow(image_hand)

        # Outermost point (just top left rect point for now)
        # TODO: find the one that is actually still pixel of the hand
        M = (x, y)
        distance = ((centroid[0] - M[0]) ** 2.0 + (centroid[1] - M[1]) ** 2.0) ** 0.5
        stepcount = 30
        step = distance / stepcount

        # Image density function (returns 0 or 1 for each pixel, but never twice "1" for the same)
        # TODO: rename
        tmp = numpy.copy(binary)
        tmp[binary == 255] = 1

        def D(posx, posy):
            # out of bounds?
            if posx < 0 or posx > tmp.shape[0] or posy < 0 or posy > tmp.shape[1]:
                return 0

            # already returned or simply 0?
            if tmp[posy, posx] == 0:
                return 0

            # never return "1" twice
            tmp[posy, posx] = 0
            return 1

        # compute integral / sum over all angles
        sums = [0 for _ in xrange(stepcount)]
        for i in xrange(stepcount):
            radius = i*step

            # sum over angles
            for k in xrange(360):
                phi = numpy.deg2rad(k)
                # to polar coordinates, offset to centroid center
                xpos = radius * math.cos(phi) + centroid[0]
                ypos = radius * math.sin(phi) + centroid[1]
                sums[i] += D(xpos, ypos)  # TODO: make sure no pixel is added twice

        t = numpy.arange(0.0, distance, step)
        t = t[:len(sums)]
        self.next_plot()
        print sums
        pyplot.plot(t, sums)

        print "Discrete Wavelet transform"
        a, b = self.discrete_wavelet_transform(sums)
        self.vectors.append([a, b, filename])

        # now do wavelet transform on the resulting 1d function
        # cwtmatr = self.wavelet_transform(sums)
        # print "Wavelet result matrix shape: %s" % str(cwtmatr.shape)

        self.show()

    def train_som(self):

        training_data = [v[0] for v in self.vectors]
        from minisom import MiniSom
        size = len(training_data[0])
        self.som = MiniSom(10, 10, size, sigma=0.3, learning_rate=0.5)
        print "Training SOM..."
        self.som.train_random(training_data, 100)
        print "...ready!"

    def use_som(self):

        print "Distance map", self.som.distance_map()
        training_data = [v[0] for v in self.vectors]
        for v in training_data:
            print self.som.winner(v)


    def find_contours(self, binary):
        copy = numpy.copy(binary)
        contours, _ = cv2.findContours(copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def wavelet_transform(self, data):

        # TODO: perform discrete wavelet transform instead

        sig = numpy.copy(data)
        widths = numpy.arange(1, len(data))
        cwtmatr = signal.cwt(sig, signal.ricker, widths)

        self.next_plot()
        pyplot.imshow(cwtmatr, extent=[-1, 1, 1, 31], cmap='PRGn', aspect='auto',
                    vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())

        return cwtmatr

    def normalize_vectors(self):
        """
        Bring all vectors in self.vectors to the same length
        """
        maxlength = 0
        for v in self.vectors:
            maxlength = len(v) if len(v) > maxlength else maxlength

        self.vectors = [v + [0 for _ in xrange(maxlength - len(v))] for v in self.vectors]

    def discrete_wavelet_transform(self, data):
        return pywt.dwt(data, 'db1')

    def largest_bounding_rect(self, contours):

        x, y, w, h = 0, 0, 0, 0

        for cnt in contours:
            rect = cv2.boundingRect(cnt)
            _1, _2, width, height = rect
            if width*height > w*h:
                x, y, w, h = rect

        return x, y, w, h

    def next_plot(self):
        self.plotCount += 1
        pyplot.subplot(3, 3, self.plotCount)

    def imshow(self, img):
        self.next_plot()
        pyplot.imshow(img, pyplot.cm.gray)

    def show(self):

        if self.save:
            # TODO: image filename
            self.filenumber += 1
            filename = "output/out-%d.png" % self.filenumber
            print "Writing %s ..." % filename
            pyplot.savefig(filename)

        if self.showwindow:
            pyplot.show()


if __name__ == '__main__':

    files = glob.glob1('images', '*2.pgm')
    tool = HandProcessor(showwindow=False, save=True)

    print "%d files with dark background found" % len(files)

    for filename in files[:5]:

        filename = "images/%s" % filename
        image = read_pgm(filename, byteorder='<')
        tool.process(image, filename)

    # tool.normalize_vectors()
    tool.train_som()
    tool.use_som()