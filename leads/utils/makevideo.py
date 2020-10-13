import cv2
import os
from skimage.io.collection import alphanumeric_key

def png_to_video_cv2(image_folder, video_name, fps=7,
                     extension='.png', scaling=1,
                     encoding=0x7634706d):
    '''
    Directory containing images
    '''
    images = [img for img in os.listdir(image_folder) if img.endswith(extension)]
    images = sorted(images, key=alphanumeric_key)
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    new_height = scaling * height
    new_width = scaling * width

    video = cv2.VideoWriter(video_name, encoding, fps, (new_width,new_height))
    for image in images:
        frame = cv2.imread(os.path.join(image_folder, image))
        frame = cv2.resize(frame, (new_width, new_height), interpolation = cv2.INTER_NEAREST) 
        video.write(frame)
    cv2.destroyAllWindows()
    video.release()
    return