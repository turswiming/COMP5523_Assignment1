from PIL import Image # pillow package
import numpy as np
from scipy import ndimage
from PIL import ImageDraw

def read_img_as_array(file):
    '''read image and convert it to numpy array with dtype np.float64'''
    img = Image.open(file)
    arr = np.asarray(img, dtype=np.float64)
    return arr

def save_array_as_img(arr, file):
    
    # make sure arr falls in [0,255]
    min, max = arr.min(), arr.max()
    if min < 0 or max > 255:
        arr = (arr - min)/(max-min)*255

    arr = arr.astype(np.uint8)
    img = Image.fromarray(arr)
    img.save(file)

def show_array_as_img(arr):
    min, max = arr.min(), arr.max()
    if min < 0 or max > 255:  # make sure arr falls in [0,255]
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

def sobel(arr):
    '''Apply sobel operator on arr and return the result.'''
    # TODO: Please complete this function.
    # your code here
    sobel_kernel_x = np.array([
        [-1, 0, 1], 
        [-2, 0, 2], 
        [-1, 0, 1]])
    sobel_kernel_y = np.array([
        [1, 2, 1], 
        [0, 0, 0], 
        [-1, -2, -1]])
    s = 3
    padding_img = np.zeros((img.shape[0]+s-1, img.shape[1]+s-1,3))
    padding_size = s//2
    padding_img[padding_size:-padding_size, padding_size:-padding_size] = img
    #full padding
    for i in range(padding_size):
        padding_img[padding_size:padding_img.shape[0]-padding_size, i] = img[:, 0]
        padding_img[padding_size:padding_img.shape[0]-padding_size, -i-1] = img[:, -1]
        padding_img[i, padding_size:padding_img.shape[1]-padding_size] = img[0, :]
        padding_img[-i-1, padding_size:padding_img.shape[1]-padding_size] = img[-1, :]
    #corner padding
    for i in range(padding_size):
        for j in range(padding_size):
            padding_img[i, j] = img[0, 0]
            padding_img[-i-1, -j-1] = img[-1, -1]
            padding_img[-i-1, j] = img[-1, 0]
            padding_img[i, -j-1] = img[0, -1]
    Gx = np.zeros(arr.shape)
    Gy = np.zeros(arr.shape)
    for i in range(1, arr.shape[0]-1):
        for j in range(1, arr.shape[1]-1):
            Gx[i,j] = np.sum(arr[i-1:i+2, j-1:j+2] * sobel_kernel_x)
            Gy[i,j] = np.sum(arr[i-1:i+2, j-1:j+2] * sobel_kernel_y)
    G = np.sqrt(Gx**2 + Gy**2)
    return G, Gx, Gy

def nonmax_suppress(G, Gx, Gy):
    '''Suppress non-max value along direction perpendicular to the edge.'''
    assert G.shape == Gx.shape
    assert G.shape == Gy.shape
    # TODO: Please complete this function.
    # your code here
    theta_array = np.arctan2(Gy, Gx) * 180 / np.pi #degree
    suppressed_G = G.copy()
    for i in range(1, G.shape[0]-1):
        for j in range(1, G.shape[1]-1):
            theta = theta_array[i,j]
            if 22.5<theta<67.5 or -112.5<theta<-67.5:
                N1 = G[i-1, j+1]
                N2 = G[i+1, j-1]
            elif 67.5<theta<112.5 or -67.5<theta<-22.5:
                N1 = G[i-1, j]
                N2 = G[i+1, j]
            elif 112.5<theta<157.5 or -22.5<theta<22.5:
                N1 = G[i-1, j-1]
                N2 = G[i+1, j+1]
            else:
                N1 = G[i, j-1]
                N2 = G[i, j+1]
            if G[i,j]<N1 or G[i,j]<N2:
                suppressed_G[i,j] = 0
                

    return suppressed_G

def thresholding(G, t):
    '''Binarize G according threshold t'''
    G_binary = G.copy()
    G_binary[G_binary<t] = 0
    G_binary[G_binary>=t] = 255
    return G_binary

def hysteresis_thresholding(G, low, high):
    '''Binarize G according threshold low and high'''
    # TODO: Please complete this function.
    # your code here
    G_low = thresholding(G, low)
    G_high = thresholding(G, high)
    visited = np.zeros(G.shape)
    def visit(G, visited, x, y):
        stark = [(x,y)]
        while stark:
            x, y = stark.pop()
            for i in range(x-1, x+2):
                for j in range(y-1, y+2):
                    if i<0 or j<0 or i>=G.shape[0] or j>=G.shape[1]:
                        continue
                    if G[i,j] == 255 and visited[i,j] == 0:
                        visited[i,j] = 1
                        stark.append((i,j))


    for i in range(1, G.shape[0]-1):
        for j in range(1, G.shape[1]-1):
            if G_high[i,j] == 255:
                visit(G_low, visited, i, j)
    G_hyst = visited*255
    return G_low, G_high, G_hyst

def hough(G_hyst):
    '''Return Hough transform of G'''
    rho_max = int(np.hypot(G_hyst.shape[0], G_hyst.shape[1]))
    theta_max = 180
    scale_rho = 1
    scale_theta = 5
    hough_space = np.zeros((2 * rho_max*scale_rho, theta_max*scale_theta))

    for i in range(G_hyst.shape[0]):
        for j in range(G_hyst.shape[1]):
            if G_hyst[i, j] == 255:
                for theta in range(theta_max*scale_theta):
                    theta = theta / scale_theta
                    rho = i * np.cos(theta * np.pi / 180) + j * np.sin(theta * np.pi / 180)
                    rho_index = rho + rho_max  # Shift rho to positive index
                    rho_index = int(rho_index * scale_rho)
                    if 0 <= rho_index < 2 * rho_max*scale_rho:
                        hough_space[rho_index, int(theta*scale_theta)] += 1

    hough_voting_path = "data/2.6_hough.jpg"
    save_array_as_img(hough_space, hough_voting_path)


    def get_local_max(img):
        local_max = np.zeros(img.shape)
        for i in range(1, img.shape[0] - 1):
            for j in range(1, img.shape[1] - 1):
                if img[i, j] > img[i - 1, j - 1] and img[i, j] > img[i - 1, j] and img[i, j] > img[i - 1, j + 1] and \
                   img[i, j] > img[i, j - 1] and img[i, j] > img[i, j + 1] and img[i, j] > img[i + 1, j - 1] and \
                   img[i, j] > img[i + 1, j] and img[i, j] > img[i + 1, j + 1]:
                    local_max[i, j] = img[i, j]
        return local_max

    local_max = get_local_max(hough_space)

    top_line_size = 30
    top_lines = []
    for _ in range(top_line_size):
        rho, theta = np.unravel_index(local_max.argmax(), local_max.shape)
        top_lines.append((rho/scale_rho - rho_max, theta/scale_theta))  # Adjust rho back to original range
        local_max[rho, theta] = 0
    colored_hough = np.zeros((hough_space.shape[0], hough_space.shape[1], 3))
    colored_hough[:, :, 0] = hough_space

    im = Image.fromarray((colored_hough * 255 / np.max(colored_hough)).astype(np.uint8))
    draw = ImageDraw.Draw(im)
    for rho, theta in top_lines:
        draw.point((theta*scale_theta, (rho + rho_max)*scale_rho))  # Adjust rho for drawing
    im.show()
    return top_lines
    

if __name__ == '__main__':
    input_path = 'data/road.jpeg'
    img = read_img_as_array(input_path)
    gray_path = 'data/2.1_gray.jpg'
    G_path = "data/2.3_G.jpg"
    Gx_path = "data/2.3_G_x.jpg"
    Gy_path = "data/2.3_G_y.jpg"
    supressed_G_path = "data/2.4_supress.jpg"
    edgemap_low_path = "data/2.5_edgemap_low.jpg"
    edgemap_high_path = "data/2.5_edgemap_high.jpg"
    edgemap_hyst_path = "data/2.5_edgemap.jpg"
    gray = rgb2gray(img)
    gray = ndimage.gaussian_filter(gray, sigma=1)
    save_array_as_img(gray, gray_path)
    G, Gx, Gy = sobel(gray)
    save_array_as_img(G, G_path)
    save_array_as_img(Gx, Gx_path)
    save_array_as_img(Gy, Gy_path)
    supressed_array = nonmax_suppress(G, Gx, Gy)
    save_array_as_img(supressed_array, supressed_G_path)
    low, high, hyst = hysteresis_thresholding(G, 100, 200)
    save_array_as_img(low, edgemap_low_path)
    save_array_as_img(high, edgemap_high_path)
    save_array_as_img(hyst, edgemap_hyst_path)
    top_lines = hough(hyst)
    print(top_lines)
    top_lines_ab = []
    for rho, theta in top_lines:
        theta_rad = np.deg2rad(theta)
        a = -np.cos(theta_rad) / np.sin(theta_rad)
        a = 1/a
        b = rho / np.cos(theta_rad)
        top_lines_ab.append((a, b))
    im = Image.open(input_path)
    draw = ImageDraw.Draw(im)
    for a, b in top_lines_ab:
        x0 = 0
        y0 = a * x0 + b
        x1 = im.size[0]
        y1 = a * x1 + b
        draw.line((x0, y0, x1, y1), fill=128)
    im.show()
    #save the drawn image
    im.save("data/2.7_detection_result.jpg")
    
