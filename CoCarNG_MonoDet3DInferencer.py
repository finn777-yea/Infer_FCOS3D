import pickle
import tempfile
import numpy as np
from mmdet3d.apis import MonoDet3DInferencer
from mmdet3d.apis import inference_mono_3d_detector
from mmdet3d.visualization import Det3DLocalVisualizer
from mmdet3d.visualization.vis_utils import proj_camera_bbox3d_to_img
from mmdet3d.apis.inference import init_model
from mmdet3d.structures import CameraInstance3DBoxes
import torch

# manually compute an autofactor to change the bbox for COCAR
def get_autofactor(camera_matrix, factor=1.15):
    autofactor = 1266.417203046554 / (factor * camera_matrix[0][0])
    return autofactor


def modify_anotation(image_path, camera_matrix):
    infos = {}
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
    return infos_file

'''

def detection_array_to_camera_instance_3d_boxes(detection_array, cam_instance_box):
        new_camera_instance_3d_bboxe = cam_instance_box.clone()
        boxes_data = []
        for detection in detection_array.detections:
            euler_angles = euler_from_quaternion([detection.pose.pose.orientation.x, detection.pose.pose.orientation.y, detection.pose.pose.orientation.z, detection.pose.pose.orientation.w])
            boxes_data.append([detection.pose.pose.position.x, 
                               detection.pose.pose.position.y + (detection.bounding_box_size.vector.y/2.0), 
                               detection.pose.pose.position.z, 
                               detection.bounding_box_size.vector.x, 
                               detection.bounding_box_size.vector.y, 
                               detection.bounding_box_size.vector.z, 
                               euler_angles[1]])
        new_camera_instance_3d_bboxe.tensor = torch.tensor(boxes_data, dtype=torch.float32)
        return new_camera_instance_3d_bboxe

'''

def main():
    # image_path: str = '/home/ws/uqmfs/mmdetection3d/samples_01/front_medium_1699456120-947280865.jpg'
    image_path: str = '/home/ws/uqmfs/mmdetection3d/samples_01/front_medium_1695968890-599103868.jpg'

    config_path = "/home/ws/uqmfs/mmdetection3d/configs/fcos3d/fcos3d_r101-caffe-dcn_fpn_head-gn_8xb2-1x_nus-mono3d_finetune.py"
    checkpoint_path = "/home/ws/uqmfs/mmdetection3d/weights/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune_20210717_095645-8d806dc2.pth"

    inferencer = MonoDet3DInferencer(config_path, checkpoint_path)

    # https://ids-git.fzi.de/interne-projekte/robots/cocar-nextgen/cocar-nextgen-camera/-/blob/main/config/calibration/cocar-nextgen-front-medium-calib.yaml?ref_type=heads (or from /camera_info topic)
    camera_matrix = np.array((1614.61267, 0.0, 975.26354, 0.0, 0.0, 1694.19946, 589.81466, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)).reshape((4,4))
    projection_matrix = np.array((1614.61267 / 2., 0.0, 975.26354 / 2., 0.0, 0.0, 1694.19946 / 2., 589.81466 / 2., 0.0, 0.0, 0.0, 1.0, 0.0)).reshape((3,4))

    infos_file = modify_anotation(image_path, camera_matrix)
    autofactor = get_autofactor(projection_matrix)

    infer_input = dict(img=image_path, infos=infos_file.name)
    
    result: dict = inferencer(
        infer_input,
        show=False,
        return_datasamples=True,
        return_vis=True
    )

    
    preds = result['predictions'][0].pred_instances_3d

    for i, pred_bbox in enumerate(preds.bboxes_3d):
        pos_x, pos_y, pos_z = pred_bbox[0:3]
        size_x, size_y = pred_bbox[3:5]
        pos_y_adjusted = pos_y + 0.5 * size_y
        preds.bboxes_3d[i][1] = pos_y_adjusted
    
    # TODO: check in which line the infer_input is modified
    infer_input = dict(img=image_path, infos=infos_file.name)
    vis_input = inferencer._inputs_to_list(infer_input)
    inferencer.visualize(vis_input, preds, show=True)


if __name__ == "__main__":
    main()