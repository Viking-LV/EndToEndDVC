import os
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

IMG_EXTENSIONS = [
    '.jpg',
    '.JPG',
    '.jpeg',
    '.JPEG',
    '.png',
    '.PNG',
    '.ppm',
    '.PPM',
    '.bmp',
    '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def choice_train(root):
    images = []
    for video in os.listdir(root):
        for frame in os.listdir(root + "/" + video):
            for name in os.listdir(root + "/" + video + "/" + frame):
                images.append('{}'.format(video + "/" + frame + "/" + name))
    return images


def yuv2rgb(yuvfilename, W, H, startframe, totalframe, show=False, out=False):
    # 从第startframe（含）开始读（0-based），共读totalframe帧
    arr = np.zeros((totalframe, H, W, 3), np.uint8)
    path = yuvfilename.replace("vimeo_triplet", "vimeo_H265")[:42]
    if not os.path.exists(path):
        os.mkdir(path)

    plt.ion()
    yuvfilename = yuvfilename[:53]
    with open(yuvfilename, 'rb') as fp:
        seekPixels = startframe * H * W * 3 // 2
        fp.seek(8 * seekPixels)  # 跳过前startframe帧

        for i in range(totalframe):
            print(i)
            oneframe_I420 = np.zeros((H * 3 // 2, W), np.uint8)
            for j in range(H * 3 // 2):
                for k in range(W):
                    oneframe_I420[j, k] = int.from_bytes(fp.read(1), byteorder='little', signed=False)
            oneframe_RGB = cv2.cvtColor(oneframe_I420, cv2.COLOR_YUV2RGB_I420)
            if show:
                plt.imshow(oneframe_RGB)
                plt.show()
                plt.pause(0.001)
            if out:
                # outname = path.split("\\")[6] + '_' + str(startframe + i + 1).zfill(5) + '.png'
                outname = yuvfilename.split('/')[-1].replace("yuv","png")
                # outname = yuvfilename[:-4] + '_' +  str(startframe + i + 1).zfill(5) + '.png'
                cv2.imwrite(path + "/" + outname, oneframe_RGB[:, :, ::-1])
            arr[i] = oneframe_RGB
    return arr

def psnr(img1, img2):
    mse = np.mean( (img1/255. - img2/255.) ** 2 )
    # mse = np.mean((img1/1.0 - img2/1.0) ** 2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
if __name__ == '__main__':
    root = "/home/whut4/lvting/dataset/vimeo_triplet/sequences"
    rootH265 = "/home/whut4/lvting/dataset/vimeo_H265/sequences"
    img = choice_train(root)
    for i in range(len(img)):
        path = root + '/' + img[i]
        pathH265 = rootH265 + '/' + img[i]
        imgOrg = cv2.imread(path)
        imgH265 = cv2.imread(pathH265)

        if(imgH265.shape != imgOrg.shape):
        # if(psnr(imgOrg, imgH265)<30):
            print(pathH265)
