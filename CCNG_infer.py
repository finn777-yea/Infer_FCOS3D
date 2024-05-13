import pickle
import tempfile
import numpy as np
from mmdet3d.apis import inference

config_path = "/home/ws/uqmfs/mmdetection3d/configs/fcos3d/fcos3d_r101-caffe-dcn_fpn_head-gn_8xb2-1x_nus-mono3d_finetune.py"
checkpoint_path = "/home/ws/uqmfs/mmdetection3d/weights/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune_20210717_095645-8d806dc2.pth"

# https://ids-git.fzi.de/interne-projekte/robots/cocar-nextgen/cocar-nextgen-camera/-/blob/main/config/calibration/cocar-nextgen-front-medium-calib.yaml?ref_type=heads (or from /camera_info topic)
camera_matrix = np.array((1614.61267 / 2, 0.0, 975.26354 / 2, 0.0, 0.0, 1694.19946 / 2, 589.81466 / 2, 0.0, 0.0, 0.0, 1.0, 0.0)).reshape((3,4))

# camera_matrix = np.array((1614.61267, 0.0, 975.26354, 0.0, 0.0, 1694.19946, 589.81466, 0.0, 0.0, 0.0, 1.0, 0.0)).reshape((3,4))    # the original cam matrix

# manually compute an autofactor to change the bbox for COCAR
factor = 1.15
autofactor = 1266.417203046554 / (factor * camera_matrix[0][0])


camera_matrix = np.vstack((camera_matrix, np.array([0, 0, 0, 1])))

# image_path: str = '/home/ws/uqmfs/mmdetection3d/samples_01/front_medium_1699456120-947280865.jpg'
image_path: str = '/home/ws/uqmfs/mmdetection3d/samples_01/front_medium_1695968890-599103868.jpg'
infos={
    'data_list': [
        {
            'images': {
                'CAM2': {
                    'img_path': image_path,
                    'cam2img': camera_matrix,
                    'lidar2img': np.array([]),
                    'lidar2cam': np.array([]),
                }
            }
        }
    ]
}

infos_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pkl')
pickle.dump(infos, infos_file)
infos_file.close()

# initialize the model and use it for inference
fcos_model = inference.init_model(config_path, checkpoint_path)
results = inference.inference_mono_3d_detector(fcos_model, image_path, infos_file.name, 'CAM2')

# extract the bboxes information out of the inference result
pred_bboxes = results.pred_instances_3d.bboxes_3d.tensor.cpu().numpy()
print(f"The number of the predicted bboxes: {pred_bboxes.shape[0]}")
pred_bbox = pred_bboxes[0]
pos_x, pos_y, pos_z = pred_bbox[0:3]
size_x, size_y = pred_bbox[3:5]
yaw = pred_bbox[6]


# modify the bboxes with the autofactor
pos_x = pos_x / autofactor
pos_y = pos_y / autofactor
pos_y = pos_y - ((1/autofactor)/2.0 * size_y)

print(pos_x, pos_y)

