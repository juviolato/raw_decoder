import cv2 as cv
import rawpy
import numpy as np

RED = 2
GREEN = 1
BLUE = 0
PERCENTILE = 99.75
GAMMA = 1.25

cfa = np.zeros((2, 2, 3), dtype=int)
cfa[:, :, RED]    = [[1, 0], [0, 0]]
cfa[:, :, GREEN]  = [[0, 1], [1, 0]]
cfa[:, :, BLUE]   = [[0, 0], [0, 1]]


def pixel_color(raw_image, i, j, color):
  cfa_i = i % 2
  cfa_j = j % 2

  if cfa[cfa_i, cfa_j, color] == 1:
    return raw_image[i, j]
  else:
    count = 0
    value = 0
    for i_offset in range(-1, 2):
      for j_offset in range(-1, 2):
        if cfa_i + i_offset == -1:
          final_i = 1 
        elif cfa_i + i_offset == 2:
          final_i = 0
        else:
          final_i = cfa_i + i_offset

        if cfa_j + j_offset == -1:
          final_j = 1
        elif cfa_j + j_offset == 2:
          final_j = 0
        else:
          final_j = cfa_j + j_offset
        
        if cfa[final_i, final_j, color] == 1:
          if i + i_offset >= 0 and j + j_offset >= 0:
            if i + i_offset < raw_image.shape[0] and j + j_offset < raw_image.shape[1]:
              count = count + 1
              value = value + raw_image[i + i_offset][j + j_offset]

    return value/count  


def bilinear_interpolation(raw_image):
  height = raw_image.shape[0]
  width = raw_image.shape[1]

  image = np.zeros((height, width, 3), dtype=int)
  
  for i in range(height):
    for j in range(width):
      image[i, j, RED]    = pixel_color(raw_image, i, j, RED)
      image[i, j, GREEN]  = pixel_color(raw_image, i, j, GREEN)
      image[i, j, BLUE]   = pixel_color(raw_image, i, j, BLUE)

  return image


def white_balance(image):
  height = image.shape[0]
  width = image.shape[1]
  balanced = np.zeros((height, width, 3), dtype=int)

  # find max intensity value for each color channel
  #max_R = 0
  #max_G = 0
  #max_B = 0
  #for i in range(height):
  #  for j in range(width):
  #    img_luminosity = 0.2126 * image[i, j, RED] + 0.7152 * image[i, j, GREEN] + 0.0722 * image[i, j, BLUE]
  #    max_luminosity = 0.2126 * max_R + 0.7152 * max_G + 0.0722 * max_B
  #    if img_luminosity > max_luminosity:
  #      max_R = image[i, j, RED]
  #      max_G = image[i, j, GREEN]
  #      max_B = image[i, j, BLUE]

  # extract a percentile as the max value for each channel
  red_pixels = image[:, :, RED]
  max_R = np.percentile(red_pixels, PERCENTILE)
  green_pixels = image[:, :, GREEN]
  max_G = np.percentile(green_pixels, PERCENTILE)
  blue_pixels = image[:, :, BLUE]
  max_B = np.percentile(blue_pixels, PERCENTILE)

  for i in range(height):
    for j in range(width):
      balanced[i, j, RED]   = min(image[i, j, RED] * 255 / max_R, 255)
      balanced[i, j, GREEN] = min(image[i, j, GREEN] * 255 / max_G, 255)
      balanced[i, j, BLUE]  = min(image[i, j, BLUE] * 255 / max_B, 255)
  
  return balanced


def gamma_correction(image):
  height = image.shape[0]
  width = image.shape[1]

  corrected = np.zeros((height, width, 3), dtype=int)
  for i in range(height):
    for j in range(width):
      corrected[i, j, RED]    = ((image[i, j, RED] / 255) ** (1 / GAMMA)) * 255
      corrected[i, j, GREEN]  = ((image[i, j, GREEN] / 255) ** (1 / GAMMA)) * 255
      corrected[i, j, BLUE]   = ((image[i, j, BLUE] / 255) ** (1 / GAMMA)) * 255
    
  return corrected


if __name__ == '__main__':
  path = 'scene.dng'

  with rawpy.imread(path) as raw:
    raw_image = raw.raw_image.copy()
  
  image = bilinear_interpolation(raw_image)
  #image = cv.imread(path)
  image_8u = np.ndarray((image.shape[0], image.shape[1], 3))
  image_8u = cv.normalize(image, image_8u, 0, 255, cv.NORM_MINMAX, cv.CV_8UC3)
  cv.imwrite("demosaic.jpg", image_8u)

  wimage = white_balance(image_8u)
  wimage_8u = np.ndarray((wimage.shape[0], wimage.shape[1], 3))
  wimage_8u = cv.normalize(wimage, wimage_8u, 0, 255, cv.NORM_MINMAX, cv.CV_8UC3)
  cv.imwrite("white_balance_" + str(PERCENTILE) + ".jpg", wimage_8u)

  gimage = gamma_correction(wimage_8u)
  gimage_8u = np.ndarray((gimage.shape[0], gimage.shape[1], 3))
  gimage_8u = cv.normalize(gimage, gimage_8u, 0, 255, cv.NORM_MINMAX, cv.CV_8UC3)
  cv.imwrite("gamma_" + str(PERCENTILE) + "_" + str(GAMMA) + ".jpg", gimage_8u)
  GAMMA = 2
  gimage = gamma_correction(wimage_8u)
  gimage_8u = np.ndarray((gimage.shape[0], gimage.shape[1], 3))
  gimage_8u = cv.normalize(gimage, gimage_8u, 0, 255, cv.NORM_MINMAX, cv.CV_8UC3)
  cv.imwrite("gamma_" + str(PERCENTILE) + "_" + str(GAMMA) + ".jpg", gimage_8u)