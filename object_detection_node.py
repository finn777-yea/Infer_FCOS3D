S#!/usr/bin/env python3
import sys
import os
import ruamel.yaml
import numpy as np
import time
import copy
from threading import Lock

#ROS
import rospy
from tf.transformations import quaternion_from_euler, euler_from_quaternion
import cv2
from cv_bridge import CvBridge, CvBridgeError
import message_filters

# ROS Messages
from sensor_msgs.msg import Image as Image_msg
from sensor_msgs.msg import CameraInfo as CameraInfo_msg
from geometry_msgs.msg import Quaternion
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
from tks_driving_stack_msgs.msg import Detection3D, Detection3DArray, ScoreClassTuple

import rospkg

import torch
from torch.profiler import profile, record_function, ProfilerActivity

class Bbox:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = min(x1, x2)
        self.x2 = max(x1, x2)
        self.y1 = min(y1, y2)
        self.y2 = max(y1, y2)
    
    def construct_from_xywh(self, x, y, w, h):
        self.x1 = x - (w / 2)
        self.x2 = x + (w / 2)
        self.y1 = y - (h / 2)
        self.y2 = y + (h / 2)

    def intersect(self, bbox):
        x1 = max(self.x1, bbox.x1)
        y1 = max(self.y1, bbox.y1)
        x2 = min(self.x2, bbox.x2)
        y2 = min(self.y2, bbox.y2)
        width = (x2-x1)
        height = (y2-y1)
        if (width < 0.0 or height < 0.0):
            return 0.0
        area_overlap = width * height
        return area_overlap
    def iou(self, bbox):
        area_overlap = self.intersect(bbox)
        area_a = (self.x2 - self.x1) * (self.y2 - self.y1)
        area_b = (bbox.x2 - bbox.x1) * (bbox.y2 - bbox.y1)
        area_combined = area_a + area_b - area_overlap

        # RATIO OF AREA OF OVERLAP OVER COMBINED AREA
        iou = area_overlap / (area_combined+1e-5)
        return iou
    def toTorch(self):
        return torch.Tensor([x1,y1,x2,y2])

    def x1y1x2y2_to_xywh(self):
        w = (self.x2 - self.x1)
        h = (self.y2 - self.y1)
        center_x = self.x1 + (w / 2.0)
        center_y = self.y1 + (h / 2.0)
        return center_x, center_y, w, h

    
    def print_box(self):
      print(self.x1,self.x2,self.y1,self.y2)

class Bbox_3d:
  def __init__(self, x1, y1, z1, x2, y2, z2):
    self.x1 = min(x1, x2)
    self.x2 = max(x1, x2)
    self.y1 = min(y1, y2)
    self.y2 = max(y1, y2)
    self.z1 = min(z1, z2)
    self.z2 = max(z1, z2)
  
  def project_3d_point_to_2d(self, points, g_p_sync):
      point = np.matmul(g_p_sync, points)
      point[0,:] = point[0,:] / point[2,:]
      point[1,:] = point[1,:] / point[2,:]
      return point[0:2,:]
  
  def project_3d_box_to_2d(self, g_p_sync):
      points = self.project_3d_point_to_2d(np.array([[self.x1,self.x1,self.x1,self.x1,self.x2,self.x2,self.x2,self.x2],[self.y1,self.y1, self.y2,self.y2, self.y1,self.y1, self.y2,self.y2],[self.z1, self.z2,self.z1, self.z2,self.z1, self.z2,self.z1, self.z2]], dtype=np.float32), g_p_sync)

      u1 = min(points[0,:])
      u2 = max(points[0,:])

      v1 = min(points[1,:])
      v2 = max(points[1,:])

      box_2d = Bbox(u1, v1, u2, v2)
      return box_2d
  
class object_detection:
    def __init__(self):
        rospack = rospkg.RosPack()
        mappath = os.path.join(rospack.get_path("tks_driving_stack_msgs"), "maps/nuScenes_benchmark.yaml")
        yaml = ruamel.yaml.YAML()
        with open(mappath, 'r') as stream:
            nuScenes_map = yaml.load(stream)
        self.msg_to_cls_dict = nuScenes_map['classes']
        self.cls_to_msg_dict = {v: k for k, v in self.msg_to_cls_dict.items()}
        self.output_to_cls = {0:'car', 
                                1:'truck', 
                                2:'trailer', 
                                3:'bus', 
                                4:'construction_vehicle', 
                                5:'bicycle', 
                                6:'motorcycle', 
                                7:'pedestrian', 
                                8:'traffic_cone', 
                                9:'barrier',
                                10:'undefined'
                                }
        self.output_to_attr = {0:'cycle.with_rider', 
                                1:'cycle.without_rider', 
                                2:'pedestrian.moving', 
                                3:'pedestrian.standing', 
                                4:'pedestrian.sitting_lying_down', 
                                5:'vehicle.moving', 
                                6:'vehicle.parked', 
                                7:'vehicle.stopped', 
                                8:'undefined'
                                }

        self.mmdetection3d_path = rospy.get_param('~mmdetection3d_path', '~/mmdetection3d')
        self.config_file = rospy.get_param('~model_config', os.path.join(self.mmdetection3d_path, "configs/fcos3d/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune_webcam.py"))
        self.checkpoint_file = rospy.get_param('~checkpoint_file', os.path.join(self.mmdetection3d_path, "checkpoints/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune_20210717_095645-8d806dc2.pth"))
        self.publish_image = rospy.get_param('~publish_image', True)
        self.score_thr = rospy.get_param('~score_thr', 0.0)
        self.nms_thr = rospy.get_param('~nms_thr', 0.0)
        self.hard_nms = rospy.get_param('~hard_nms', False) #use false
        self.nms_2d_thr = rospy.get_param('~nms_2d_thr', 0.90)
        self.publish_rate = rospy.get_param('~publish_rate', 15.0) # max rate feedforward
        self.factor = rospy.get_param('~factor', 1.0) # manual projection factor
        self.gpu = rospy.get_param('~gpu', 0)
        self.profile = rospy.get_param('~profile', False)

        self.bridge = CvBridge()
        self.mutex = Lock() 


        self.img_msg = None         #latest received image
        self.camera_info = None     #latest received camera info
        self.img_received = False   # True if self.img_msg is not none
        self.img_data_sync = None       #currently used cv image as numpy array in main thread
        self.img_header_sync = None     #currently used image header in main thread
        self.camera_info_sync = None    #currently used camera info used in main thread
        self.g_p_sync = None            #currently used camera projection in main thread
        
        # Load model
        sys.path.append(self.mmdetection3d_path)
        from mmdet3d.apis import init_model

        from mmdet3d.visualization import Det3DLocalVisualizer
        from mmdet3d.structures import CameraInstance3DBoxes

        devicestr = 'cuda:'+str(int(self.gpu))
        rospy.loginfo("Inititalizing Neural Network on GPU " + devicestr)
        self.device=torch.device('cuda')
        starttime=time.time()
        #cfg_options = dict(nms_thr = self.nms_thr)
        self.model_uncompiled = init_model(self.config_file, self.checkpoint_file, device=devicestr, nms_thr=self.nms_thr)
        #self.model_uncompiled = self.model_uncompiled.to(memory_format=torch.channels_last)
        #self.model = init_model(self.config_file, self.checkpoint_file, device="cpu", nms_thr=self.nms_thr)
        
        print("loading took ", time.time() - starttime)
        rospy.loginfo("Neural Network initialized")
        
        print(torch.cuda.get_device_capability())
        
        self.model = torch.compile(self.model_uncompiled, mode="max-autotune")
        print("Compiling took ", time.time()-starttime)
        
        #self.img_sub = message_filters.Subscriber("image_in", Image_msg, queue_size=1, buff_size=2**32)
        #self.info_sub = message_filters.Subscriber("cam_info_in", CameraInfo_msg, queue_size=1)
        #ts = message_filters.TimeSynchronizer([self.img_sub, self.info_sub], 10)
        #ts.registerCallback(self.multi_cb)
        self.img_sub = rospy.Subscriber("image_in", Image_msg, self.image_cb, queue_size=1, buff_size=2**32)
        self.info_sub = rospy.Subscriber("cam_info_in", CameraInfo_msg, self.info_cb, queue_size=1)

        self.detections_pub = rospy.Publisher("~detections", Detection3DArray, queue_size=1)
        self.detections_2d_pub = rospy.Publisher("~detections_2d", Detection2DArray, queue_size=1)
        
        if (self.publish_image):
            self.detections_image_pub = rospy.Publisher("~output_img/detections_vis", Image_msg, queue_size=1)
            self.detections_caminfo_pub = rospy.Publisher("~output_img/camera_info", CameraInfo_msg, queue_size=1)

            self.visualizer = Det3DLocalVisualizer(save_dir="/dev/null")

        self.input_latency = 0.0
        self.input_latency_sum = 0.0
        self.pub_img_latency = 0.0
        self.pub_img_latency_sum = 0.0

        self.final_latency = 0.0
        self.final_latency_sum = 0.0
        self.total_ffs = 0
        self.total_ff_latency = 0.0

        self.nms_2d_latency = 0.0
        self.nms_2d_latency_sum = 0.0
        
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

        rospy.loginfo("Publishers and Subscribers initialized")
        
        rate = rospy.Rate(self.publish_rate)

        while not self.img_received and (self.camera_info is not None) and not rospy.is_shutdown():

            rospy.logwarn("No initial image received")
            try:
                rate.sleep()
            except rospy.exceptions.ROSTimeMovedBackwardsException as e:
                continue
            except Exception as e:
                print(e)
                return
        rospy.loginfo("Main loop starts")

        prof = None
        profile = False
        if profile:
            prof = torch.profiler.profile(schedule=torch.profiler.schedule(wait=20, warmup=20, active=1, repeat=1), on_trace_ready=torch.profiler.tensorboard_trace_handler('/log/base/ncwh_fp16'), record_shapes=True, with_stack=True)
            prof.start()

        while not rospy.is_shutdown():    
            if profile:
                prof.step()
                if self.total_ffs == 20 + 20 + 1:
                    prof.stop()
            self.feed_forward()
            #todo remove this after finding out good params
            self.score_thr = rospy.get_param('~score_thr', 0.0)
            self.nms_thr = rospy.get_param('~nms_thr', 0.0)
            self.hard_nms = rospy.get_param('~hard_nms', False)
            self.nms_2d_thr = rospy.get_param('~nms_2d_thr', 0.90)
            
            try:
                rate.sleep()
            except rospy.exceptions.ROSTimeMovedBackwardsException as e:
                self.input_latency = 0.0
                self.input_latency_sum = 0.0
                self.final_latency = 0.0
                self.final_latency_sum = 0.0
                self.total_ffs = 0
                self.total_ff_latency = 0.0
                self.nms_2d_latency = 0.0
                self.nms_2d_latency_sum = 0.0
                continue
            except Exception as e:
                print(e)
                return
            
        #except KeyboardInterrupt:
        #    print("Keyboard Interrupt Detected: Shutting down")
            #sys.exit()
    
    # only store messages in global variables in callback
    def multi_cb(self, data_image, data_info):
        if self.mutex.acquire(blocking=False):
            self.camera_info = data_info
            self.img_msg = data_image
            self.mutex.release()
        self.img_received = True

    def image_cb(self, data_image):
        if self.mutex.acquire(blocking=False):
            self.img_msg = data_image
            self.img_received = True
            self.mutex.release()
    def info_cb(self, data_info):
        self.camera_info = data_info
        
    def get_newest_data(self):
        if self.mutex.acquire(blocking=True, timeout=1/self.publish_rate):
            img_msg = self.img_msg
            self.camera_info_sync = self.camera_info
            
            self.img_msg = None
            #self.camera_info = None #allow old cam_infos
            self.mutex.release()
        if img_msg is None or self.camera_info_sync is None: #global self.img_msg was none so no new image since last feedforward
            return False
        self.img_header_sync = img_msg.header
        if self.img_header_sync.frame_id == "":
            self.img_header_sync.frame_id = "fcos_frame"
        p = self.camera_info_sync.P
        self.g_p_sync = [[p[0], p[1], p[2]], [p[4],p[5],p[6]], [p[8], p[9], p[10]]]
        #self.g_p_sync = [[1631.61462/2.0, 0, 982.67275/2.0], [0, 1720.53406/2.0, 593.8216/2.0], [0, 0, 1]] #MRZ Simulation parameters
        #temp_cam_info.P = [1631.61462/2.0, 0, 982.67275/2.0, 0.0 , 0, 1720.53406/2.0, 593.8216/2.0, 0.0, 0, 0, 1/2.0, 0] #MRZ Simulation parameters

        if (img_msg.header.stamp != self.camera_info_sync.header.stamp):
            rospy.logwarn_throttle(10.0, "Camera_Info and Image do not have the same timestamp")

        self.input_latency = round((rospy.Time.now() - img_msg.header.stamp).to_sec(), 3)
        self.input_latency_sum += self.input_latency
        return img_msg
    
    def convert_image(self, img_msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
        except CvBridgeError as e:
            print(e)
            return False
        self.img_data_sync = np.asarray(cv_image)
        return True

        

    def feed_forward(self):
        startff = rospy.Time.now()

        if (not (img_msg := self.get_newest_data())):
            rospy.loginfo("No new image in queue")
            return
        
        if not self.convert_image(img_msg):
            rospy.logwarn("Could not convert input image")
            return
        
        conversion_time = rospy.Time.now()
        conversion_duration = conversion_time-startff
        
        from mmdet3d.apis import live_inference_mono_3d_detector

        if self.g_p_sync is None or self.img_data_sync is None or self.img_header_sync is None:
            return

        result = live_inference_mono_3d_detector(self.model, self.img_data_sync, self.g_p_sync)
        
        model_ff_duration = round((rospy.Time.now() - conversion_time).to_sec(), 3)
        published_results = self.publish_result(result)
        
        if (self.publish_image):
            self.draw_result(result, published_results)
        self.final_latency = round((rospy.Time.now() - startff).to_sec(), 3)
        
        if (self.final_latency >= 0):
            self.total_ffs = self.total_ffs + 1
            self.final_latency_sum += self.final_latency
            self.total_ff_latency += model_ff_duration
            self.pub_img_latency_sum += self.pub_img_latency
            self.nms_2d_latency_sum += self.nms_2d_latency

        else:
            self.input_latency_sum -= self.input_latency

        if(self.total_ffs > 0 and self.final_latency > 0):
            rospy.loginfo(f'Latency input: current, avg: {self.input_latency:.3f}, {self.input_latency_sum/self.total_ffs:.3f}')
            rospy.loginfo(f'Latency Model Feedforward: current, avg {model_ff_duration:.3f},  {self.total_ff_latency/self.total_ffs:.3f}')
            rospy.loginfo(f'Latency NMS 2D: current, avg {self.nms_2d_latency:.3f},  {self.nms_2d_latency_sum/self.total_ffs:.3f}')
            rospy.loginfo(f'Latency publish image: current, avg {self.pub_img_latency:.3f},  {self.pub_img_latency_sum/self.total_ffs:.3f}')
            rospy.loginfo(f'Latency ROS Feed-Forward: current, avg: {self.final_latency:.3f}, {1.0/self.final_latency:.1f}, {self.final_latency_sum/self.total_ffs:.3f}')
            rospy.loginfo("----------------")

    def draw_result(self, result, published_results):
        img_publish_time = rospy.Time.now()

        #self.visualizer.set_image(self.img_data_sync)
        overlay = self.img_data_sync.copy()

        boxes_to_draw = self.detection_array_to_camera_instance_3d_boxes(published_results, result.pred_instances_3d.bboxes_3d)
        corners2d = self.visualizer.proj_camera_bbox3d_to_img(boxes_to_draw, {'cam2img': self.g_p_sync})
        
        lines_verts_idx = [0, 1, 2, 3, 7, 6, 5, 4, 0, 3, 7, 4, 5, 1, 2, 6]
        lines_verts = corners2d[:, lines_verts_idx, :]
        front_polys = corners2d[:, 4:, :]
        back_polys = corners2d[:, :4, :]
            
        for box in front_polys:
            tmp_box = box.reshape((-1,1,2))
            cv2.fillPoly(overlay, np.int32([tmp_box]), color=(0,255,255))
        for box in back_polys:
            tmp_box = box.reshape((-1,1,2))
            cv2.fillPoly(overlay, np.int32([tmp_box]), color=(0,255,255))

        for box in lines_verts:
            tmp_box = box.reshape((-1,1,2))
            cv2.polylines(self.img_data_sync, np.int32([tmp_box]), False, (0,255,255), thickness=3)

        alpha = 0.4
        img_to_draw = cv2.addWeighted(overlay, alpha, self.img_data_sync, 1 - alpha, 0)
        
        #self.visualizer.draw_proj_bboxes_3d(boxes_to_draw, {'cam2img': self.g_p_sync})
        #drawn_image = self.visualizer.get_image()

        self.pub_img_latency = round((rospy.Time.now() - img_publish_time).to_sec(), 3)

        image_message = self.bridge.cv2_to_imgmsg(img_to_draw, encoding="bgr8")
        image_message.header = self.img_header_sync
        temp_cam_info = self.camera_info_sync
        temp_cam_info.header = self.img_header_sync

        self.detections_image_pub.publish(image_message)
        self.detections_caminfo_pub.publish(temp_cam_info)

    def detection_array_to_camera_instance_3d_boxes(self, detection_array, cam_instance_box):
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


    def publish_result(self, result):
        pred_bboxes = result.pred_instances_3d.bboxes_3d.tensor.cpu().numpy()
        pred_scores = result.pred_instances_3d.scores_3d.cpu().numpy()
        pred_labels = result.pred_instances_3d.labels_3d.cpu().numpy()
        pred_label_scores = np.ones(pred_labels.shape)
        pred_attrs = result.pred_instances_3d.attr_labels.cpu().numpy()
        pred_attrs_scores = np.ones(pred_attrs.shape)

        if self.score_thr > 0:
            inds = pred_scores > self.score_thr
            #print("score thr pruned", pred_bboxes.shape[0] - np.sum(inds), "boxes,", int((np.sum(inds)/pred_bboxes.shape[0])*100), "%", np.sum(inds), "survive")

            pred_bboxes = pred_bboxes[inds]
            pred_scores = pred_scores[inds]
            pred_labels = pred_labels[inds]
            pred_label_scores = pred_label_scores[inds]
            pred_attrs = pred_attrs[inds]

        detected_objects = Detection3DArray()
        detected_objects.header = self.img_header_sync
        detected_objects.classification_map_identifier = 5

        for i in range(len(pred_bboxes)):
            detected_objects.detections.append(self.generate_3D_Object_Msg(pred_bboxes[i], pred_label_scores[i], pred_attrs_scores[i]))
            detected_objects.detections[i].tuples = self.generate_score_tuples(pred_scores[i], pred_labels[i], pred_attrs[i])
        
        if self.hard_nms: #make false
            detected_objects.detections = self.det_nms(detected_objects.detections)
        if False: #2d NMS
            detected_objects.detections = self.nms_2d(detected_objects.detections, self.nms_2d_thr)
        
        self.detections_pub.publish(detected_objects)
        convert2dstart = rospy.Time.now()
        self.detections_2d_pub.publish(self.convert_to_2d_detection(detected_objects))
        return detected_objects
        #print("conversion",round((rospy.Time.now() - convert2dstart).to_sec(), 3))

    def convert_detection_3d_to_box3d(self, detection_3d):
        box_3d = Bbox_3d(detection_3d.pose.pose.position.x -(detection_3d.bounding_box_size.vector.x / 2.0), 
                    detection_3d.pose.pose.position.y -(detection_3d.bounding_box_size.vector.y / 2.0), 
                    detection_3d.pose.pose.position.z -(detection_3d.bounding_box_size.vector.z / 2.0), 
                    detection_3d.pose.pose.position.x +(detection_3d.bounding_box_size.vector.x / 2.0), 
                    detection_3d.pose.pose.position.y +(detection_3d.bounding_box_size.vector.y / 2.0), 
                    detection_3d.pose.pose.position.z +(detection_3d.bounding_box_size.vector.z / 2.0)) 
        return box_3d

    def convert_to_2d_detection(self, detections_3d):
        detection_array_msg = Detection2DArray()
        detection_array_msg.header = detections_3d.header
        for detection_3d in detections_3d.detections:
            tmp_detection_2d = Detection2D()
            tmp_detection_2d.header = detections_3d.header
            box_3d = self.convert_detection_3d_to_box3d(detection_3d)
            box_2d = box_3d.project_3d_box_to_2d(np.array(self.g_p_sync))
            x,y,w,h = box_2d.x1y1x2y2_to_xywh()

            tmp_detection_2d.bbox.center.x = x
            tmp_detection_2d.bbox.center.y = y
            tmp_detection_2d.bbox.size_x = w
            tmp_detection_2d.bbox.size_y = h

            for tuple_single in detection_3d.tuples:
                hypothesis = ObjectHypothesisWithPose()
                hypothesis.score = tuple_single.score
                hypothesis.id = tuple_single.classification[0]
                hypothesis.pose.pose.position.x = x
                hypothesis.pose.pose.position.y = y
                hypothesis.pose.pose.position.z = 0

                tmp_detection_2d.results.append(hypothesis)
                
            detection_array_msg.detections.append(tmp_detection_2d)
        return detection_array_msg
    #proof of concept, todo replace with faster nms function
    def nms_2d(self, detections, nms_2d_thr):
        start_nms = rospy.Time.now()
        total_boxes_init = 0.0
        total_boxes_proj = 0.0
        total_iou_calc = 0.0
        total_which_box_del = 0.0
        total_del_boxes = 0.0
        

        if (nms_2d_thr > 1.0):
            self.nms_2d_latency = round((rospy.Time.now() - start_nms).to_sec(), 3)
            return detections

        ind = [] #index of boxes marked for deletion
        for i in range(len(detections)):
            for j in range(len(detections)):
                if (i == j) or (i in ind) or (j in ind):
                    pass
                else:
                    start_boxes_init = rospy.Time.now()
                    box_3d_i = Bbox_3d(detections[i].pose.pose.position.x -(detections[i].bounding_box_size.vector.x / 2.0), 
                                       detections[i].pose.pose.position.y -(detections[i].bounding_box_size.vector.y / 2.0), 
                                       detections[i].pose.pose.position.z -(detections[i].bounding_box_size.vector.z / 2.0), 
                                       detections[i].pose.pose.position.x +(detections[i].bounding_box_size.vector.x / 2.0), 
                                       detections[i].pose.pose.position.y +(detections[i].bounding_box_size.vector.y / 2.0), 
                                       detections[i].pose.pose.position.z +(detections[i].bounding_box_size.vector.z / 2.0)) 

                    box_3d_j = Bbox_3d(detections[j].pose.pose.position.x -(detections[j].bounding_box_size.vector.x / 2.0), 
                                       detections[j].pose.pose.position.y -(detections[j].bounding_box_size.vector.y / 2.0), 
                                       detections[j].pose.pose.position.z -(detections[j].bounding_box_size.vector.z / 2.0), 
                                       detections[j].pose.pose.position.x +(detections[j].bounding_box_size.vector.x / 2.0), 
                                       detections[j].pose.pose.position.y +(detections[j].bounding_box_size.vector.y / 2.0), 
                                       detections[j].pose.pose.position.z +(detections[j].bounding_box_size.vector.z / 2.0))
                    total_boxes_init += (rospy.Time.now() - start_boxes_init).to_sec()
                    start_boxes_proj = rospy.Time.now()

                    box_i = box_3d_i.project_3d_box_to_2d(np.array(self.g_p_sync))
                    box_j = box_3d_j.project_3d_box_to_2d(np.array(self.g_p_sync))

                    total_boxes_proj += (rospy.Time.now() - start_boxes_proj).to_sec()
                    start_iou = rospy.Time.now()

                    iou = box_i.iou(box_j)
                    total_iou_calc += (rospy.Time.now() - start_iou).to_sec()
                    start_which_box = rospy.Time.now()
                    if iou >= nms_2d_thr:
                        if (detections[i].tuples[0].score < detections[j].tuples[0].score):
                            ind.append(i)
                            total_which_box_del += (rospy.Time.now() - start_which_box).to_sec()
                            break
                        elif (detections[i].tuples[0].score == detections[j].tuples[0].score):
                            if (i < j):
                                ind.append(i)
                                break
                            else:
                                ind.append(j)
                        else:
                            ind.append(j)
                    total_which_box_del += (rospy.Time.now() - start_which_box).to_sec()
        start_box_del = rospy.Time.now()
        for index in sorted(set(ind), reverse=True):
            del detections[index]
        total_del_boxes += (rospy.Time.now()-start_box_del).to_sec()
        self.nms_2d_latency = round((rospy.Time.now() - start_nms).to_sec(), 3)
        print("Python 2D NMS is slow: ",total_boxes_init, total_boxes_proj, total_iou_calc, total_which_box_del, total_del_boxes)
        return detections

    #bev nms, slow, dont use increase nms_thr instead.
    def det_nms(self, detections):
        ind = []
        for i in range(len(detections)):
            for j in range(len(detections)):
                if (i == j) or (i in ind):
                    pass
                else:
                    box_i = Bbox(detections[i].pose.pose.position.x -(detections[i].bounding_box_size.vector.x / 2.0), 
                    detections[i].pose.pose.position.z - (detections[i].bounding_box_size.vector.z / 2.0), 

                    detections[i].pose.pose.position.x + (detections[i].bounding_box_size.vector.x / 2.0), 
                    detections[i].pose.pose.position.z + (detections[i].bounding_box_size.vector.z / 2.0))


                    box_j = Bbox(detections[j].pose.pose.position.x -(detections[j].bounding_box_size.vector.x / 2.0), 
                    detections[j].pose.pose.position.z - (detections[j].bounding_box_size.vector.z / 2.0), 

                    detections[j].pose.pose.position.x + (detections[j].bounding_box_size.vector.x / 2.0), 
                    detections[j].pose.pose.position.z + (detections[j].bounding_box_size.vector.z / 2.0))

                    iou = box_i.intersect(box_j)
                    if iou > 0.0:
                        if (detections[i].tuples[0].score < detections[j].tuples[0].score):
                            ind.append(i)
                            break
                        elif (detections[i].tuples[0].score == detections[j].tuples[0].score):
                            if (i < j):
                                ind.append(i)
                                break
                            else:
                                ind.append(j)
                        else:
                            ind.append(j)
        for index in sorted(set(ind), reverse=True):
            del detections[index]
        return detections

    def generate_3D_Object_Msg(self, pred_bbox, pred_label_scores, pred_attr_scores):    
        object3d = Detection3D()
        object3d.pose.pose.position.x = pred_bbox[0]
        object3d.pose.pose.position.y = pred_bbox[1]
        object3d.pose.pose.position.z = pred_bbox[2]

        #q = quaternion_from_euler(pred_bbox[8], pred_bbox[6], pred_bbox[7]) #in case of large pitch/roll
        q = quaternion_from_euler(0, pred_bbox[6], 0)
        object3d.pose.pose.orientation = Quaternion(*q)
        object3d.bounding_box_size.vector.x = pred_bbox[3]
        object3d.bounding_box_size.vector.y = pred_bbox[4] 
        object3d.bounding_box_size.vector.z = pred_bbox[5]

        autofactor = 1266.417203046554 / (self.factor * self.g_p_sync[0][0])  #1266 is the focal length of nuscenes
        #print("autofactor, ", autofactor, " p0 ", self.g_p_sync[0][0], " p0 *factor ", self.factor * self.g_p_sync[0][0]) 

        object3d.pose.pose.position.x = object3d.pose.pose.position.x / autofactor
        object3d.pose.pose.position.y = object3d.pose.pose.position.y /autofactor
        object3d.pose.pose.position.z = object3d.pose.pose.position.z / autofactor
        object3d.pose.pose.position.y = object3d.pose.pose.position.y - ((1/autofactor)/2.0 * object3d.bounding_box_size.vector.y)

        return object3d

    def generate_score_tuples(self, pred_score, pred_label, pred_attr):
        tuples = []
        tmp_tuple = ScoreClassTuple()
        tmp_tuple.score = pred_score
        tmp_tuple.classification = self.output_to_msg_cls(pred_label)
        cls_string = self.output_to_cls[pred_label]
        tuples.append(tmp_tuple)
        attr_tuple = self.attr_to_msg_cls(cls_string, pred_attr, pred_score)
        if attr_tuple is not None:
            tuples.append(attr_tuple)
        return tuples

    def generate_score_tuplesold(self, pred_label_scores, pred_attrs_scores):
        tuples = []
        sorted_label_index = np.argsort(pred_label_scores)[::-1]
        for i in sorted_label_index:
            tmp_tuple = ScoreClassTuple()
            tmp_tuple.score = pred_label_scores[i]
            tmp_tuple.classification = self.output_to_msg_cls(i)
            cls_string = self.output_to_cls[i]
            tuples.append(tmp_tuple)
            for j in range(len(pred_attrs_scores)):
                attr_tuple = self.attr_to_msg_cls(cls_string, j, pred_attrs_scores[j])
                if attr_tuple is not None:
                    tuples.append(attr_tuple)
        
        return tuples

    def output_to_msg_cls(self, output):
        cls_string = self.output_to_cls[output]
        msg_key = self.cls_to_msg_dict[cls_string]
        return msg_key
    
    def attr_to_msg_cls(self, cls_string, attr_output, attr_score):
        atttr_string = self.output_to_attr[attr_output]
        if self.attr_fits_cls(cls_string, atttr_string):
            tmp_tuple = ScoreClassTuple()
            tmp_tuple.score = attr_score
            if atttr_string != "undefined":
                tmp_tuple.classification = self.cls_to_msg_dict[cls_string + "." + atttr_string.split(".")[1]]                        
            return tmp_tuple
        return None

    #Does the attribute fit to the class? The attribute with_rider doesnt make sense for the class car
    def attr_fits_cls(self, cls, attr):       
        if (cls == 'car' or cls == 'truck' or cls =='bus' or cls == 'construction_vehicle' or cls == 'trailer'):
            if (attr == 'vehicle.moving' or attr == 'vehicle.parked' or attr == 'vehicle.stopped'):
                return True
        elif (cls == 'pedestrian'):
            if (attr == 'pedestrian.moving' or attr == 'pedestrian.standing' or attr == 'pedestrian.sitting_lying_down'):
                return True
        elif (cls == 'bicycle' or cls == 'motorcycle'):
            if (attr == 'cycle.with_rider' or attr == 'cycle.without_rider'):
                return True
        return False

def main(args):
    rospy.init_node('object_detection', anonymous=True)

    detector = object_detection()
    
if __name__ == '__main__':
    try:
        main(sys.argv)
    except rospy.ROSInterruptException:
        pass