import os
import json
import cv2
import math
import numpy as np
import paddle
import time
from source.video.infer import Detector, DetectorPicoDet, PredictConfig, print_arguments, get_test_images
from source.video.keypoint_infer import KeyPoint_Detector, PredictConfig_KeyPoint
from source.video.visualize import draw_pose
from source.video.keypoint_postprocess import translate_to_ori_images
from source.video.tools import cal_ang, cross_point,Point,Calculate
import threading
KEYPOINT_SUPPORT_MODELS = {
    'HigherHRNet': 'keypoint_bottomup',
    'HRNet': 'keypoint_topdown'
}


def predict_with_given_det(image, det_res, keypoint_detector,
                           keypoint_batch_size, det_threshold,
                           keypoint_threshold, run_benchmark):
    rec_images, records, det_rects = keypoint_detector.get_person_from_rect(
        image, det_res, det_threshold)
    keypoint_vector = []
    score_vector = []
    rect_vector = det_rects
    batch_loop_cnt = math.ceil(float(len(rec_images)) / keypoint_batch_size)

    for i in range(batch_loop_cnt):
        start_index = i * keypoint_batch_size
        end_index = min((i + 1) * keypoint_batch_size, len(rec_images))
        batch_images = rec_images[start_index:end_index]
        batch_records = np.array(records[start_index:end_index])
        if run_benchmark:
            # warmup
            keypoint_result = keypoint_detector.predict(
                batch_images, keypoint_threshold, repeats=10, add_timer=False)
            # run benchmark
            keypoint_result = keypoint_detector.predict(
                batch_images, keypoint_threshold, repeats=10, add_timer=True)
        else:
            keypoint_result = keypoint_detector.predict(batch_images,
                                                        keypoint_threshold)
        orgkeypoints, scores = translate_to_ori_images(keypoint_result,
                                                       batch_records)
        keypoint_vector.append(orgkeypoints)
        score_vector.append(scores)

    keypoint_res = {}
    keypoint_res['keypoint'] = [
        np.vstack(keypoint_vector).tolist(), np.vstack(score_vector).tolist()
    ] if len(keypoint_vector) > 0 else [[], []]
    keypoint_res['bbox'] = rect_vector
    return keypoint_res

class stateManager(object):



    def __init__(self):
        '''
        左右方向不同时至少两次连续才能确认
        三次连续跳就是大跳
        action_list 状态索引顺序: '左', '右', '下', '停', '跑', '跳', '打', '其它'

        [1, 0, 0, 0, 0, 0, 0] action, jump, left, right, up, down, return
        '''

        self.reset()
        self.is_right = 1 # 1->right, 0->left
    def reset(self):
        self.current_state = [0, 0, 0, 0, 0, 0, 0]

    #根据状态获取动作
    def get_action(self, act):
        self.reset()
        if act == "squat":
            self.current_state[5] = 1
        elif act == "left":
            self.is_right = 0
            self.current_state[2] = 1
        elif act == "right":
            self.is_right = 1
            self.current_state[3] = 1
        elif act == "fire":
            self.current_state[0] = 1
        elif act == "run":
            self.current_state[0] = 1
            idx = 3 if self.is_right else 2
            self.current_state[idx] = 1
        elif act == "jump":
            self.current_state[1] = 1
        elif act == "walk":
            idx = 3 if self.is_right else 2
            self.current_state[idx] = 1

        return self.current_state

    #持续上次动作
    def get_last_action(self):
        print("send last: ", self.current_state)
        return self.current_state


class Video_Process (threading.Thread):
    def __init__(self, act_que):

        self.device = "GPU"
        self.run_mode = 'fluid'
        self.keypoint_batch_size = 1
        self.trt_min_shape = 1
        self.trt_max_shape = 1280
        self.trt_opt_shape = 640
        self.trt_calib_mode = False
        self.cpu_threads = 1
        self.enable_mkldnn = False
        self.use_dark = True
        self.det_threshold= 0.5
        self.keypoint_threshold = 0.5
        self.run_benchmark = False
        self.det_model_dir = 'models/picodet_s_192_pedestrian'
        self.pred_config = PredictConfig(self.det_model_dir)
        detector_func = 'Detector'
        if self.pred_config.arch == 'PicoDet':
            detector_func = 'DetectorPicoDet'
        self.detector = eval(detector_func)(self.pred_config,
                                       self.det_model_dir,
                                       device=self.device,
                                       run_mode=self.run_mode,
                                       trt_min_shape=self.trt_min_shape,
                                       trt_max_shape=self.trt_max_shape,
                                       trt_opt_shape=self.trt_opt_shape,
                                       trt_calib_mode=self.trt_calib_mode,
                                       cpu_threads=self.cpu_threads,
                                       enable_mkldnn=self.enable_mkldnn)


        self.keypoint_model_dir = 'models/tinypose_128x96'
        self.pred_config = PredictConfig_KeyPoint(self.keypoint_model_dir)
        assert KEYPOINT_SUPPORT_MODELS[
                   self.pred_config.arch] == 'keypoint_topdown', 'Detection-Keypoint unite inference only supports topdown models.'
        self.topdown_keypoint_detector = KeyPoint_Detector(
            self.pred_config,
            self.keypoint_model_dir,
            device=self.device,
            run_mode=self.run_mode,
            batch_size=self.keypoint_batch_size,
            trt_min_shape=self.trt_min_shape,
            trt_max_shape=self.trt_max_shape,
            trt_opt_shape=self.trt_opt_shape,
            trt_calib_mode=self.trt_calib_mode,
            cpu_threads=self.cpu_threads,
            enable_mkldnn=self.enable_mkldnn,
            use_dark=self.use_dark)

        self.state_manager = stateManager()
        self.action_queue = act_que
        self.frame_action_state = None

        threading.Thread.__init__(self)

    def update_action(self, act_name):
        cur_state = self.state_manager.get_action(act_name)
        for i in range(len(cur_state)):
            if cur_state[i] > 0 :
                self.frame_action_state[i] = 1

    def send_action(self, act_name):
        self.action_queue.put(self.state_manager.get_action(act_name), True, 2)

    def run(self):
        video_file = 0 #写死
        capture = cv2.VideoCapture(video_file)
        # (480, 640, 3)

        index = 0
        store_res = []

        visual_thread = 0.4
        nose_his = None
        Left_Ankle_his = None
        Right_Ankle_his = None
        Left_Knee_his = None
        Right_Knee_his = None
        Left_Shoulder_his = None
        Right_Shoulder_his = None
        jump_list = []

        while (1):
            ret, frame = capture.read()
            if not ret:
                break
            index += 1
            # print('detect frame: %d' % (index))


            # self.frame_action_state = [0, 0, 0, 0, 0, 0, 0]
            index += 1
            # print('detect frame: %d' % (index))

            frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.detector.predict([frame2], self.det_threshold)

            keypoint_res = predict_with_given_det(
                frame2, results, self.topdown_keypoint_detector, self.keypoint_batch_size,
                self.det_threshold, self.keypoint_threshold, self.run_benchmark)

            if len(keypoint_res['keypoint'][0]):

                # 太多了，记不到，需要用到的关键点都先取出来放着
                keypoint = keypoint_res['keypoint'][0][0]  # 取出全部关键点关键点
                nose = keypoint[0]  # 鼻子
                Left_Shoulder = keypoint[5]  # 左肩
                Right_Shoulder = keypoint[6]  # 右肩
                Left_Elbow = keypoint[7]  # 左肘
                Right_Elbow = keypoint[8]  # 右肘
                Left_Wrist = keypoint[9]  # 左手腕
                Righ_Wris = keypoint[10]  # 右手腕
                Left_Hip = keypoint[11]  # 左髋
                Right_Hip = keypoint[12]  # 右髋
                Left_Knee = keypoint[13]  # 左膝盖
                Right_Knee = keypoint[14]  # 右膝盖
                Left_Ankle = keypoint[15]  # 左脚踝
                Right_Ankle = keypoint[16]  # 右脚踝

                # 下蹲 髋骨的点位于膝盖与脚之间，或者髋骨的点小于膝盖
                # print(Left_Hip[2],Right_Hip[2])
                if Left_Hip[2] > visual_thread or Right_Hip[2] > visual_thread:
                    if Left_Hip[2] > visual_thread or Right_Hip[2] > visual_thread:
                        if abs(Left_Hip[1] - Left_Knee[1]) < 30:
                            self.send_action("squat")
                        elif abs(Right_Hip[1] - Right_Knee[1]) < 30:
                            self.send_action("squat")

                    # 转向  手抬高至肩以上
                    Turn_left_angle = cal_ang(Left_Elbow[:2], Left_Shoulder[:2], Left_Hip[:2])  # 左三角角度
                    Turn_right_angle = cal_ang(Right_Elbow[:2], Right_Shoulder[:2], Right_Hip[:2])  # 右三角角度
                    if int(Turn_left_angle) > 75 and int(Turn_right_angle) < 45:
                        self.send_action("left")
                        print('left')
                    if int(Turn_right_angle) > 75 and int(Turn_left_angle) < 45:
                        self.send_action("right")
                        print('right')
                    # print("angle: ", Turn_left_angle, Turn_right_angle)

                    # 发射   # 双手交叉则进行发射
                    left_line = Left_Elbow[:2] + Left_Wrist[:2]
                    left_line = [int(x) for x in left_line]
                    right_line = Righ_Wris[:2] + Right_Elbow[:2]
                    right_line = [int(x) for x in right_line]

                    cross, a = cross_point(left_line, right_line)
                    # print(left_line,right_line)
                    # print(cross,a)
                    if cross and max(a) < max((left_line + right_line)) and min(a) > min((left_line + right_line)):
                        self.send_action("fire")

                    # 走 膝盖与膝盖的点差值，脚踝与脚踝的脚差值
                    # print(abs(Left_Knee[1]-Right_Knee[1]) , abs(Left_Ankle[1] - Right_Ankle[1]))
                    if abs(Left_Knee[1] - Right_Knee[1]) > 10 and abs(Left_Ankle[1] - Right_Ankle[1]) > 20:
                        if int(Turn_left_angle) > 20 and int(Turn_right_angle) > 20:
                            self.send_action("run")
                            print('run')

                        else:
                            self.send_action("walk")
                            print('walk')
                    # else:
                    #     return '停止'
                    # 跳跃 时序上的双脚踝的上下移动 或者 简单的隔10帧脚踝
                    # print(Left_Ankle_his ,Right_Ankle_his ,Left_Knee_his ,Right_Knee_his)
                    # print(Left_Ankle,Right_Ankle,Left_Knee,Right_Knee)
                    # print(nose)
                    tis = 3
                    tis_ = -3
                    if Left_Ankle_his is not None and Right_Ankle_his is not None and Left_Knee_his is not None and Right_Knee_his is not None and Right_Ankle_his is not None and Left_Shoulder_his is not None and nose_his is not None:
                        if (Left_Ankle[1] - Left_Ankle_his[1]) > tis and (Left_Knee[1] - Left_Knee_his[1]) > tis and (
                                Right_Ankle[1] - Right_Ankle_his[1]) > tis and (
                                Right_Knee[1] - Right_Knee_his[1]) > tis and (
                                Left_Shoulder[1] - Left_Shoulder_his[1]) > tis and (
                                Right_Ankle[1] - Right_Ankle_his[1]) > tis and (nose[1] - nose_his[1]) > tis:
                            print('tiao')
                            jump_list.append(1)
                        elif (Left_Ankle[1] - Left_Ankle_his[1]) < tis_ and (Left_Knee[1] - Left_Knee_his[1]) < tis_ and (
                                Right_Ankle[1] - Right_Ankle_his[1]) < tis_ and (
                                Right_Knee[1] - Right_Knee_his[1]) < tis_ and (
                                Left_Shoulder[1] - Left_Shoulder_his[1]) < tis_ and (
                                Right_Ankle[1] - Right_Ankle_his[1]) < tis_ and (nose[1] - nose_his[1]) < tis_:
                            print('tiao')
                            jump_list.append(1)
                        else:
                            jump_list.append(0)
                        if index % 2 == 0:
                            nose_his = nose
                            Left_Ankle_his = Left_Ankle
                            Right_Ankle_his = Right_Ankle
                            Left_Knee_his = Left_Knee
                            Right_Knee_his = Right_Knee
                            Left_Shoulder_his = Left_Shoulder
                            Right_Shoulder_his = Right_Shoulder

                        if len(jump_list) >  2 and jump_list[-2:] == [1,1]:

                            self.send_action("jump")
                            time.sleep(0.005)
                            self.send_action("jump")
                            time.sleep(0.005)
                            self.send_action("jump")
                            time.sleep(0.005)
                            self.send_action("jump")

                    else:
                        nose_his = nose
                        Left_Ankle_his = Left_Ankle
                        Right_Ankle_his = Right_Ankle
                        Left_Knee_his = Left_Knee
                        Right_Knee_his = Right_Knee
                        Left_Shoulder_his = Left_Shoulder
                        Right_Shoulder_his = Right_Shoulder

            im = draw_pose(
                frame,
                keypoint_res,
                visual_thread=self.keypoint_threshold,
                returnimg=True)

            cv2.imshow('Mask Detection', im)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
if __name__ == '__main__':
    paddle.enable_static()
    vp = Video_Process()
    vp.topdown_unite_predict_video(0)
