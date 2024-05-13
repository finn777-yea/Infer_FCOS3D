from mmdet3d.apis import MonoDet3DInferencer

CONFIG_PATH = f"/home/ws/uqmfs/mmdetection3d/configs/fcos3d/fcos3d_r101-caffe-dcn_fpn_head-gn_8xb2-1x_nus-mono3d_finetune.py"
CHECKPOINT_PATH = f"/home/ws/uqmfs/mmdetection3d/weights/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune_20210717_095645-8d806dc2.pth"

IMAGE_PATH = f"/home/ws/uqmfs/mmdetection3d/demo/data/nuscenes/n015-2018-07-24-11-22-45+0800__CAM_BACK__1532402927637525.jpg"
IMAGE_PATH_CoCarNG = "/home/ws/uqmfs/mmdetection3d/samples_01/front_medium_1695968890-554271312.jpg"
# Not doing its job with given a whole dir feeded
# IMAGE_PATH = f"/home/ws/uqmfs/mmdetection3d/demo/data/nuscenes"

ANO_PATH = f"/home/ws/uqmfs/mmdetection3d/demo/data/nuscenes/n015-2018-07-24-11-22-45+0800.pkl"

inferencer = MonoDet3DInferencer(CONFIG_PATH, CHECKPOINT_PATH)

inputs = dict(img=IMAGE_PATH_CoCarNG,
             infos=ANO_PATH
             )

inferencer(inputs, show=True)
# inferencer(inputs, show=True, out_dir='/Infer_FCOS3D/outputs')
