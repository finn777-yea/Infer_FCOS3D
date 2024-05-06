import pickle
import tempfile
import numpy as np
from mmdet3d.apis import MonoDet3DInferencer
from mmdet3d.apis import inference_mono_3d_detector
import torch


# manually compute an autofactor to change the bbox for COCAR
def get_autofactor(camera_matrix, factor=1.15):
    autofactor = 1266.417203046554 / (factor * camera_matrix[0][0])
    return autofactor

# correct/complete the form of camera matrix
def modify_cam_matrix(camera_matrix):
    camera_matrix = np.vstack((camera_matrix, np.array([0, 0, 0, 1])))
    # camera_matrix[1,1] *= 0.65  # weird hack that makes bboxes appear more realistically (at least in 2d)
    return camera_matrix


def modify_anotation(image_path, camera_matrix):
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


def main():
    # image_path: str = '/home/ws/uqmfs/mmdetection3d/samples_01/front_medium_1699456120-947280865.jpg'
    image_path: str = '/home/ws/uqmfs/mmdetection3d/samples_01/front_medium_1695968890-599103868.jpg'

    config_path = "/home/ws/uqmfs/mmdetection3d/configs/fcos3d/fcos3d_r101-caffe-dcn_fpn_head-gn_8xb2-1x_nus-mono3d_finetune.py"
    checkpoint_path = "/home/ws/uqmfs/mmdetection3d/weights/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune_20210717_095645-8d806dc2.pth"

    inferencer = MonoDet3DInferencer(config_path, checkpoint_path)

    # https://ids-git.fzi.de/interne-projekte/robots/cocar-nextgen/cocar-nextgen-camera/-/blob/main/config/calibration/cocar-nextgen-front-medium-calib.yaml?ref_type=heads (or from /camera_info topic)
    camera_matrix = np.array((1614.61267 / 2, 0.0, 975.26354 / 2, 0.0, 0.0, 1694.19946 / 2, 589.81466 / 2, 0.0, 0.0, 0.0, 1.0, 0.0)).reshape((3,4))
    camera_matrix = modify_cam_matrix(camera_matrix)

    infos_file = modify_anotation(image_path, camera_matrix)
    autofactor = get_autofactor(camera_matrix)

    # result is a dict with 2 keys: prediction and visualization
    result: dict = inferencer(
        dict(img=image_path, infos=infos_file.name),
        show=False,
        return_vis=True,
        no_save_vis=False
        # out_dir='/tmp/output',
        # img_out_dir='/tmp/output'
        # pred_out_dir='./tmp/output'
    )

    preds = result['predictions']
    
    pred_bboxes = torch.tensor(preds[0]['bboxes_3d']).cpu().numpy()
    pred_bbox = pred_bboxes[0]      # the first bbox in the prediction
    pos_x, pos_y, pos_z = pred_bbox[0:3]
    size_x, size_y = pred_bbox[3:5]
    yaw = pred_bbox[6]

    # modify the bboxes with the autofactor
    pos_x = pos_x / autofactor
    pos_y = pos_y / autofactor
    pos_y = pos_y - ((1/autofactor)/2.0 * size_y)

# TODO use the new pos_x/y/z to assign preds[0]['bboxes_3d'][0], and pass to visualize
    preds[0]['bboxes_3d'][0][0:3] = [pos_x, pos_y, pos_z]

    inferencer.visualize(inferencer._inputs_to_list(dict(img=image_path, infos=infos_file.name)), 
                         preds, show=True
                        #  show=True, return_vis=True, img_out_dir='./Infer_FCOS3D/output')
    )

if __name__ == "__main__":
    main()