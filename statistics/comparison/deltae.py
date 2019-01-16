import cv2
import numpy

def deltae(color1, color2, maxCount=255):
   """
   title::
      deltae

   description::
      This method will compute the color difference (delta E) between 
      two provided colors triplets or two provided color images.  The
      colors provided are assumed to be in sRGB color space and viewed 
      under illuminant D65.  If the provided colors are individual color
      triplets, the return value will be a scalar delta E.  If the provided
      colors are images, a delta E image and a scalar average delta E will
      be returned as a tuple.

   attributes::
      color1
         A color triplet [B,G,R] in an array-like object or an image 
         in a numpy.ndarray.
      color2
         A color triplet [B,G,R] in an array-like object or an image 
         in a numpy.ndarray.
      maxCount
         The maximum digital count value that might be contained in the 
         color triplets or the provide color images (this is not 
         necessarily the maximum value in the provided colors, but 
         rather it is the largest value that any component of the colors
         might take on).

   author::
      Carl Salvaggio

   copyright::
      Copyright (C) 2015, Rochester Institute of Technology

   license::
      GPL

   version::
      1.0.0

   disclaimer::
      This source code is provided "as is" and without warranties as to 
      performance or merchantability. The author and/or distributors of 
      this source code may have made statements about this source code. 
      Any such statements do not constitute warranties and shall not be 
      relied on by the user in deciding whether to use this source code.
      
      This source code is provided without any express or implied warranties 
      whatsoever. Because of the diversity of conditions and hardware under 
      which this source code may be used, no warranty of fitness for a 
      particular purpose is offered. The user is advised to test the source 
      code thoroughly before relying on it. The user must assume the entire 
      risk of using the source code.
   """

   # Make sure that the provided colors are numpy ndarrays, if not
   # convert them
   if type(color1).__module__ != numpy.__name__:
      c1 = numpy.asarray(color1)
   else:
      c1 = color1

   if type(color2).__module__ != numpy.__name__:
      c2 = numpy.asarray(color2)
   else:
      c2 = color2

   # Convert the provided colors to single-precision floating-point 
   # values in the range [0,1] for computation
   c1 = c1.astype(numpy.float32) / maxCount
   c2 = c2.astype(numpy.float32) / maxCount

   # Check that the dimensional shape of the provided colors are the same
   dimensions1 = numpy.shape(c1)
   dimensions2 = numpy.shape(c2)
   if dimensions1 != dimensions2:
      raise ValueError('Provided datasets must have the same shape')

   # Make sure the provided colors are presented as an image to the color
   # conversion routine
   if len(dimensions1) == 1 and dimensions1[-1] == 3:
      c1 = numpy.asarray(c1).reshape((1,1,3))
      c2 = numpy.asarray(c2).reshape((1,1,3))
   elif len(dimensions1) != 3 or dimensions1[-1] != 3:
      msg = 'Provided colors must be either a 3-element vector or 3xn array'
      raise ValueError(msg)

   # Convert provided sRGB colors to L*a*b* space
   lab1 = cv2.cvtColor(c1, cv2.COLOR_BGR2LAB)
   lab2 = cv2.cvtColor(c2, cv2.COLOR_BGR2LAB)

   # Compute the delta E for each of the provided color pairs
   dE = numpy.sqrt(numpy.sum((lab1 - lab2)**2, -1))

   # Return the delta E image and the average delta E value -or- the scalar 
   # delta E for a pair of provided color triplets
   if dE.size == 1:
      return float(dE[0, 0])
   else:
      return dE, float(numpy.mean(dE))


if __name__ == '__main__':
   import os.path
   import statistics.comparison

   color1 = [204, 127, 51]   # [B, G, R]
   color2 = [32, 200, 207]   # [B, G, R]
   dE = statistics.comparison.deltae(color1, color2)
   print('dE = {0}'.format(dE))

   home = os.path.expanduser('~')
   path = os.path.join(home, 'src', 'python', 'examples', 'data')
   filename = os.path.join(path, 'lenna.tif')
   image1 = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
   filename = os.path.join(path, 'lenna_blurred.tif')
   image2 = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
   dE, avgdE = statistics.comparison.deltae(image1, image2)
   print('Average dE = {0}'.format(avgdE))

