import numpy as np
import matplotlib.pyplot as plt

#using pil draw 3d height from image
def draw_3d(arr):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = np.arange(arr.shape[0])
    y = np.arange(arr.shape[1])
    X, Y = np.meshgrid(x, y)
    ax.plot_surface(X, Y, arr, cmap='viridis')
    plt.show()

x_Size = 100
y_Size = 100
a = np.array(
    [
        [1,2],
        [3,4]
        ])
print(a*a)

# img = np.zeros((x_Size, y_Size))
# for x in range(x_Size):
#     for y in range(y_Size):

#         vec = np.array([x-x_Size/2, y-x_Size/2])
#         vec = vec.reshape(1,2)
#         vecT = vec.T
#         lambda_array = np.array([[2,0],[0,1]])
#         val = np.dot(vec,lambda_array)
#         val = np.dot(val,vecT)
#         img[x, y] = val


# draw_3d(img)