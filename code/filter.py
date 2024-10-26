from PIL import Image # pillow package
import numpy as np
from scipy import ndimage
from typing import List, Tuple
def read_img_as_array(file):
    '''read image and convert it to numpy array with dtype np.float64'''
    img = Image.open(file)
    arr = np.asarray(img, dtype=np.float64)
    return arr

def save_array_as_img(arr, file):
    min, max = arr.min(), arr.max()
    if min < 0 or max > 255:  # make sure arr falls in [0,255]
        arr = (arr - min)/(max-min)*255

    arr = arr.astype(np.uint8)
    img = Image.fromarray(arr)
    img.save(file)

def show_array_as_img(arr, rescale='minmax'):
    
    # make sure arr falls in [0,255]
    min, max = arr.min(), arr.max()
    if min < 0 or max > 255:
        arr = (arr - min)/(max-min)*255

    arr = arr.astype(np.uint8)
    img = Image.fromarray(arr)
    img.show()

def rgb2gray(arr):
    R = arr[:, :, 0] # red channel
    G = arr[:, :, 1] # green channel
    B = arr[:, :, 2] # blue channel
    gray = 0.2989*R + 0.5870*G + 0.1140*B
    return gray

#########################################
## Please complete following functions ##
#########################################
def gaussian_filter(img:np.ndarray, s):
    '''Perform gaussian filter of size 3 x 3 to image 'arr', and return the filtered image.'''
    # TODO: Please complete this function.
    # your code here
    #add empty padding for img
    if s%2 == 0:
        raise("s must be odd")

    padding_img = np.zeros((img.shape[0]+s-1, img.shape[1]+s-1,3))
    padding_size = s//2
    padding_img[padding_size:-padding_size, padding_size:-padding_size] = img
    def generate_gaussian_core(s):
        x_wide = s
        y_wide = s
        #product a sigma 3x3 gaussian core, while the sigma is s
        gaussian_core = np.zeros((x_wide, y_wide))
        for i in range(x_wide):
            for j in range(y_wide):
                gaussian_core[i, j] = np.exp(-((i-1)**2+(j-1)**2)/(2*s**2))
        gaussian_core = gaussian_core/np.sum(gaussian_core)
        print(gaussian_core.shape)
        return gaussian_core
    gaussian_core = generate_gaussian_core(s)
    #convolution
    for channel in range(0,img.shape[2]):
        for i in range(padding_size, padding_img.shape[0]-padding_size):
            for j in range(padding_size, padding_img.shape[1]-padding_size):
                value = np.sum(gaussian_core*padding_img[i-padding_size:i+padding_size+1, j-padding_size:j+padding_size+1, channel])
                img[i-padding_size,j-padding_size,channel] = value
    arr = img
    return arr

def sharpen(img, sigma, alpha):
    '''Sharpen the image. 'sigma' is the standard deviation of Gaussian filter. 'alpha' controls how much details to add.'''
    # TODO: Please complete this function.
    # your code here
    arr = 1
    return arr

def median_filter(img, s):
    '''Perform median filter of size s x s to image 'arr', and return the filtered image.'''
    # TODO: Please complete this function.
    # your code here
    arr = 1
    return arr

if __name__ == '__main__':
    input_path = './data/rain.jpeg'
    img = read_img_as_array(input_path)
    # show_array_as_img(img)
    img = gaussian_filter(img, 31)
    show_array_as_img(img)
    #TODO: finish assignment Part I.
