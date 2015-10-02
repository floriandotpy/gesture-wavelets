__author__ = 'flo'

import glob
import re
import numpy
import cv2
import math
from matplotlib import pyplot

def read_pgm(filename, byteorder='>'):
    """Return image data from a raw PGM file as numpy array.

    Format specification: http://netpbm.sourceforge.net/doc/pgm.html

    """
    with open(filename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    return numpy.frombuffer(buffer,
                            dtype='u1' if int(maxval) < 256 else byteorder+'u2',
                            count=int(width)*int(height),
                            offset=len(header)
                            ).reshape((int(height), int(width)))

class HandProcessor(object):

    def __init__(self, showwindow=True, save=False):

        self.showwindow = showwindow
        self.save = save

        self.filenumber = 0

    def process(self, image):

        print "Processing..."

        pyplot.figure()
        self.plotCount = 0

        # 1. Raw image
        self.imshow(image)

        # 2. Binary image
        _, binary = cv2.threshold(image, 40, 255, cv2.THRESH_BINARY)
        self.imshow(binary)

        # 3. Contours

        # find contours
        copy = numpy.copy(binary)
        cs, _ = cv2.findContours(copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # draw contours
        image_contours = numpy.copy(binary)
        image_contours[:,:] = 0
        cv2.drawContours(image_contours, cs, -1, (255, 255, 255), 1)
        self.imshow(image_contours)

        # 4. Centroid
        image_moments = numpy.copy(image_contours)
        moments = cv2.moments(image_moments)
        centroid = (int(moments['m10']/moments['m00']), int(moments['m01']/moments['m00']))
        cv2.circle(image_moments, centroid, 3, (255, 255, 255), -1)

        # bounding rect
        cnt = cs[0]
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(image_moments,(x,y),(x+w,y+h),(255, 255, 255),2)
        self.imshow(image_moments)

        # slice hand
        image_hand = binary[y:y+h,x:x+w]
        centroid_hand = (centroid[0]-x, centroid[1]-y)
        cv2.circle(image_hand, centroid_hand, 3, (0, 0, 0), -1)
        self.imshow(image_hand)

        # outermost point (just top left rect point for now)
        # TODO: find the one that is actually still pixel of the hand
        M = (x, y)
        distance = ((centroid[0] - M[0]) ** 2.0 + (centroid[1] - M[1]) ** 2.0) ** 0.5
        #distance -= 2 # FIXME: that -2 is only because out of bounds
        stepcount = 30
        step = distance / stepcount

        # image density function (returns 0 or 1)
        # TODO: rename
        tmp = numpy.copy(binary)
        tmp[binary == 255] = 1

        def D(posx, posy):
            # out of bounds?
            if posx<0 or posx>tmp.shape[0] or posy<0 or posy>tmp.shape[1]:
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
                sums[i] += D(xpos, ypos) # TODO: make sure no pixel is added twice

        t = numpy.arange(0.0, distance, step)
        t = t[:len(sums)]
        self.nextPlot()
        print sums
        pyplot.plot(t, sums)

        # now do wavelet transform on the resulting 1d function
        # TODO

        self.show()

    def nextPlot(self):
        self.plotCount += 1
        pyplot.subplot(2, 3, self.plotCount)

    def imshow(self, img):
        self.nextPlot()
        pyplot.imshow(img, pyplot.cm.gray)

    def show(self):

        if self.save:
            # TODO: image
            self.filenumber += 1
            filename = "output/out-%d.png" % self.filenumber
            print "Writing %s ..." % filename
            pyplot.savefig(filename)

        if self.showwindow:
            pyplot.show()


    def cart2pol(x, y):
        rho = numpy.sqrt(x**2 + y**2)
        phi = numpy.arctan2(y, x)
        return(rho, phi)

    def pol2cart(rho, phi):
        x = rho * numpy.cos(phi)
        y = rho * numpy.sin(phi)
        return(x, y)


if __name__ == '__main__':

    files = glob.glob1('images', '*2.pgm')

    print "%d files with dark background found" % len(files)

    hp = HandProcessor(showwindow=True, save=False)

    for filename in files[:1]:

        filename = "images/%s" % filename

        # raw image
        image = read_pgm(filename, byteorder='<')

        hp.process(image)