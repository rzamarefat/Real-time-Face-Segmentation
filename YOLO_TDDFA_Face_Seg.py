from TDDFA import TDDFA
import yaml
from ultralytics import YOLO
from align_trans import warp_and_crop_face, get_reference_facial_points
from config import YOLO_TDDFA_CONFIG
import numpy as np

class YOLO_TDDFA:
    def __init__(self):
        cfg = yaml.load(open(YOLO_TDDFA_CONFIG["path_to_yaml_config"]), Loader=yaml.SafeLoader)
        self._tddfa = TDDFA(gpu_mode=False, **cfg)
        self._yolo_model = YOLO(YOLO_TDDFA_CONFIG["path_to_yolo_ckpt"])
        self.yolo_confidence_threshold = YOLO_TDDFA_CONFIG["yolo_confidence_threshold"] 

    def _detect(self, image):
        faces = []
        yaws = []

        results = self._yolo_model(image)
        boxes = results[0].boxes

        converted_boxes = []
        for box in boxes:
            if box.conf.item() > self.yolo_confidence_threshold:
                top_left_x = int(box.xyxy.tolist()[0][0])
                top_left_y = int(box.xyxy.tolist()[0][1])
                bottom_right_x = int(box.xyxy.tolist()[0][2])
                bottom_right_y = int(box.xyxy.tolist()[0][3])

                converted_boxes.append([top_left_x, top_left_y, bottom_right_x, bottom_right_y])
            
        param_lst, roi_box_lst = self._tddfa(image, converted_boxes)
        
        ver_lst = self._tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=True)

        return ver_lst
        
    def _extract_faces(self, img, points):
        out_face = np.zeros_like(img)
        hull_out_holder = []

        for pt in points:
            c_point = []
            for x, y in zip(pt[0, ::6].tolist(), pt[1, ::6].tolist()):
                c_point.append([int(x), int(y)])
            
            # converted_points.append(c_point)

        
            c_point = np.array(c_point)
            
            
            
            hull_index = cv2.convexHull(c_point)
            hull_out_holder.append(hull_index)
            
        for h in hull_out_holder:
            feature_mask = np.zeros((img.shape[0], img.shape[1]))    
            cv2.fillConvexPoly(feature_mask, h, 1)
            feature_mask = feature_mask.astype(bool)
            
            out_face[feature_mask] = img[feature_mask]
        
        return out_face

    def segment_face(self, img):
        points = self._detect(img)
        
        out_faces = self._extract_faces(img, points)

        return out_faces
        

if __name__ == "__main__":
    import cv2

    yolo_tddfa = YOLO_TDDFA()

    

    


    cap = cv2.VideoCapture('/home/mehran/rezamarefat/YOLO_TDDFA_Face_Seg/bradpitt.mp4')
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    counter = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        # out_faces_img = yolo_tddfa.segment_face(frame)
        cv2.imwrite(f"/home/mehran/rezamarefat/YOLO_TDDFA_Face_Seg/video_ou_main/{str(counter).zfill(5)}.jpg", frame)
        counter += 1
        print(counter)
   
    
