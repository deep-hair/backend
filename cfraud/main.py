import cv2
from detectron2.data import MetadataCatalog
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
from tqdm import tqdm

setup_logger()

from load_detectron2 import load_predictor

if __name__ == '__main__':

    out_vid = cv2.VideoWriter(
        'detectron_MH.avi',
        cv2.VideoWriter_fourcc(*'MJPG'),
        15.,
        (768, 576))

    cap = cv2.VideoCapture('../clear_test1.mov')
    cv2.startWindowThread()

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    predictor, cfg = load_predictor()
    print(total)
    for i in tqdm(range(1,total,50)):
        cap.set(1,i)
        ret, frame = cap.read()
        outputs = predictor(frame)
        v = Visualizer(frame[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.namedWindow("preview")
        cv2.imshow("prewiew", out.get_image()[:, :, ::-1])
        cv2.waitKey(1)
        out_vid.write(out.get_image()[:, :, ::-1].astype('uint8'))

    cap.release()
    cv2.destroyAllWindows()