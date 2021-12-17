
from sparse_recon.sparse_deconv import sparse_deconv
from skimage import io
from matplotlib import pyplot as plt

if __name__ == '__main__':
    im = io.imread(r"C:\Users\romanbarth\Desktop\C2-test__pos1_analysis__x221-y134-l73-w132-a0_roi.tif")
    im = im[0,0:70,0:70]
    plt.imshow(im, cmap = 'gray')
    plt.show()

    img_recon = sparse_deconv(im, [5,5])
    plt.imshow(img_recon / img_recon.max() * 255, cmap = 'gray')
    plt.show()