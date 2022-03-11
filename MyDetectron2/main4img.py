import cv2
import os

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

import cv2
from detectron2.data import MetadataCatalog
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
from tqdm import tqdm

setup_logger()

from load_detectron2 import load_predictor

if __name__ == '__main__':

    #URL = 'http://81.143.203.215:83/view/view.shtml?id=375&imagePath=%2Faxis-media%2Fmedia.amp%3Fvideocodec%3Dh264&size=1&gotopresetname=Home&camera=1'
    images = load_images_from_folder('imgs/raw_imgs')
    predictor, cfg = load_predictor()
    for i,frame in tqdm(enumerate(images)):
        frame = cv2.resize(frame,(640, 480))
        outputs = predictor(frame)
        v = Visualizer(frame[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imwrite('imgs/treated_imgs/img_{0}.jpg'.format(i),out.get_image()[:,:,::-1])

    cv2.destroyAllWindows()