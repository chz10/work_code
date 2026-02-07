# -*- coding: utf-8 -*-
import os
import re
import numpy as np
import warnings
import json
import pandas as pd
from loguru import logger
from tqdm import tqdm
from collections import Counter
from datetime import datetime
import math
import multiprocessing as mp
import argparse
import traceback
import sys
warnings.filterwarnings('ignore')

logger.add("missing_files351.log", encoding="utf-8", rotation="5 MB", retention="10 days", level="ERROR")
# def load_txt(path):
#     values = []
#     if not os.path.exists(path):
#         logger.error(f"缺失文件: {path}")
#         return values
#     with open(path, 'r', encoding='utf-8') as txt_pd:
#         for index,line in enumerate(txt_pd,start=1):
#             try:
#                 data = json.loads(line.strip())
#                 values.append(data)
#             except json.JSONDecodeError:
#                 print(f"*********无法解析  {path} 文件的第{index}行")
#     return values

def load_txt(path):
    values = {}
    if not os.path.exists(path):
        logger.error(f"缺失文件: {path}")
        return values
    with open(path, 'r', encoding='utf-8') as txt_pd:
        for index, line in enumerate(txt_pd,start=1):
            try:
                data = json.loads(line.strip())
                if "Oem" in data:
                    frame = int(data["Oem"]["u64FrameId"])
                    values[frame] = data
                else:
                    try:
                        frame, value = next(iter(data.items()))
                        frame = int(frame)
                        values[frame] = value
                    except:
                        values[index] = data

                # values.append(data)
            except json.JSONDecodeError:
                print(f"*********无法解析  {path} 文件的第{index}行")
    return values


class KalmanFilter(object):
    def __init__(self, F, H, Q, R):
        self.F = F  # 状态转移矩阵
        self.H = H  # 测量矩阵
        self.Q = Q  # 过程噪声协方差矩阵
        self.R = R  # 测量噪声方差矩阵
        self.x = None  # 状态估计
        self.P = None  # 状态协方差矩阵
        self.is_init = False

    def init(self, x0, P0):
        self.x = x0  # 初始化状态估计
        self.P = P0  # 初始化状态协方差矩阵
        self.is_init = True

    def predict(self):
        self.x = np.dot(self.F, self.x)  # 预测步骤，状态转移方程
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q  # 更新状态协方差矩阵

    def update(self, measurement):
        y = measurement - np.dot(self.H, self.x)  # 计算测量残差
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R  # 计算协方差矩阵
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))  # 计算卡尔曼增益
        self.x = self.x + np.dot(K, y)  # 更新状态估计
        self.P = np.dot((np.eye(self.x.shape[0]) - np.dot(K, self.H)), self.P)  # 更新状态协方差矩阵


class CVKF1(KalmanFilter):
    def __init__(self):
        F = np.array([[1, 1], [0, 1]])
        H = np.array([[1, 0]])
        Q = np.array([[1, 0], [0, 1]])
        R = np.array([[1]])
        super(CVKF1, self).__init__(F, H, Q, R)


class RadarObject(object):
    def __init__(self, x, y, prob, vx, vy, rid):
        self.x = x
        self.y = y
        self.prob = prob
        self.vx = vx
        self.vy = vy
        self.tid = None
        self.save_data = None
        self.rid = rid


class VisionObject(object):
    def __init__(self, tid, conf, w, h, x, y, vx, vy, angle, frame, type):
        self.tid = tid
        self.conf = conf
        self.w = w
        self.h = h
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.angle = angle
        self.frame = frame
        self.type = type
        self.save_data = None


class Track(object):
    def __init__(self, tid, radar_object, vision_object, save_data, J3_object=None, reVeh_object=None):
        self.tid = tid
        self.radar_data = [radar_object]
        self.vision_data = [vision_object]
        self.x_model = CVKF1()
        self.y_model = CVKF1()
        self.vx_model = CVKF1()
        self.save_data = [save_data]
        self.J3_data = [J3_object]
        self.reVeh_data = [reVeh_object]


class RadarVisionHybrid(object):
    def __init__(self, vision_file, vision_keyword='_arcsoft_obj.txt', mode=True, map_relate=None):
        self.vision_keyword = vision_keyword
        self.mode = mode  # 将 mode 存储为实例变量
        self.map_relate = map_relate  # 将 map_relate 存储为实例变量
        self.tracks = dict()
        self.vision_data = load_txt(vision_file)
        self.radar_data = load_txt(vision_file.replace(vision_keyword, '_ars408.txt')) if mode else load_txt(map_relate[os.path.basename(vision_file)].replace(vision_keyword, '_ars408.txt'))
        self.corner_radar_data = load_txt(vision_file.replace(vision_keyword, '_eq4_obj.txt')) if mode else load_txt(map_relate[os.path.basename(vision_file)].replace(vision_keyword, '_eq4_obj.txt'))
        self.J3_data = load_txt(vision_file.replace(vision_keyword, '_J3_obj.txt')) if mode else load_txt(map_relate[os.path.basename(vision_file)].replace(vision_keyword, '_J3_obj.txt'))
        self.pts_data = load_txt(vision_file.replace(vision_keyword, '.txt')) if mode else load_txt(map_relate[os.path.basename(vision_file)].replace(vision_keyword, '.txt'))
        self.Signal_data = load_txt(vision_file.replace(vision_keyword, '_Signal.txt')) if mode else load_txt(map_relate[os.path.basename(vision_file)].replace(vision_keyword, '_Signal.txt'))

        self.arc_reVeh_data = load_txt(vision_file.replace(vision_keyword, '_arcsoft_obj.txt')) if mode else load_txt(map_relate[os.path.basename(vision_file)].replace(vision_keyword, '_arcsoft_obj.txt'))

        self.path = os.path.dirname(vision_file)
        self.filename = os.path.basename(vision_file)
        self.mapping = dict()
        self.mapping_J3 = dict()
        self.mapping_reVeh = dict()

    def update(self, tid, radar_object, vision_object, J3_object=None, reVeh_object=None):
        if radar_object:
            save_data = {**vision_object.save_data, **radar_object.save_data}
        else:
            save_data = vision_object.save_data
        if J3_object:
            save_data.update(J3_object.save_data)
        if reVeh_object:
            save_data.update(reVeh_object.save_data)

        if tid in self.tracks.keys():
            track = self.tracks[tid]
            track.radar_data.append(radar_object)
            track.vision_data.append(vision_object)
            track.J3_data.append(J3_object)
            track.reVeh_data.append(reVeh_object)
            track.save_data.append(save_data)
        else:
            self.tracks[tid] = Track(tid, radar_object, vision_object, save_data, J3_object, reVeh_object)

        track = self.tracks[tid]
        if radar_object:
            if track.x_model.is_init is False:
                P0 = np.array([[1, 0], [0, 1]])
                x0 = np.array([[radar_object.x], [0]])
                y0 = np.array([[radar_object.y], [0]])
                vx0 = np.array([[radar_object.vx], [0]])
                track.x_model.init(x0, P0)
                track.y_model.init(y0, P0)
                track.vx_model.init(vx0, P0)
            else:
                track.x_model.update(np.array([[radar_object.x]]))
                track.y_model.update(np.array([[radar_object.y]]))
                track.vx_model.update(np.array([[radar_object.vx]]))

    def predict(self):
        for tid in self.tracks.keys():
            track = self.tracks[tid]
            if track.x_model.is_init:
                track.x_model.predict()
                track.y_model.predict()
                track.vx_model.predict()

    def cal_dists(self, radar_objects, vision_objects):

        m = len(radar_objects)
        n = len(vision_objects)
        dists = np.zeros((m, n))

        for i in range(m):
            for j in range(n):
                tid = vision_objects[j].tid
                track = self.tracks[tid] if tid in self.tracks.keys() else None

                rv_x_dist = abs(radar_objects[i].x - vision_objects[j].x)
                rv_y_dist = abs(radar_objects[i].y - vision_objects[j].y)

                vision = radar_objects[i]
                conf = getattr(vision, "prob", None)
                if conf is None:
                    conf = getattr(vision, "conf", 1.0)

                r_prob_dist = (1 - conf)
                rv_vx_dist = abs(radar_objects[i].vx - vision_objects[j].vx)
                r_x_dist = r_y_dist = 0
                x_std = y_std = vx_std = 0
                r_vx_dist = 0
                if track is not None and track.x_model.is_init:
                    pred_x = track.x_model.x[0]
                    pred_y = track.y_model.x[0]
                    pred_vx = track.vx_model.x[0]
                    r_x_dist = abs(radar_objects[i].x - pred_x)
                    r_y_dist = abs(radar_objects[i].y - pred_y)
                    r_vx_dist = abs(radar_objects[i].vx - pred_vx)

                # 提高y轴权重为x轴3倍
                dists[i][j] = 2 * rv_x_dist + 2 * rv_y_dist * 3 + 5 * r_prob_dist + \
                              rv_vx_dist + r_x_dist + r_y_dist + r_vx_dist
                # dists[i][j] = 2 * rv_x_dist + 2 * rv_y_dist * 3 + 5 * r_prob_dist + rv_vx_dist    # 去掉预测位置权重，防止跳帧间隔太久时，预测值过大异常
                gate_x = max(3, vision_objects[j].x * 0.1)
                gate_y = max(3, vision_objects[j].y * 0.1)
                # gate = max(3, radar_objects[i].prob * 10)

                # if rv_x_dist > gate_x or vision_objects[j].y * radar_objects[i].y < -0.1 or rv_y_dist > gate_y:
                if rv_x_dist > gate_x or rv_y_dist > gate_y:    # 去掉y值相反条件
                    dists[i][j] = float('inf')

        return dists

    def cal_dists_mapping(self, radar_objects, vision_objects, map_type=None):
        # 通过利用已有映射关系，进一步优化雷达目标和视觉目标之间的匹配
        m = len(radar_objects)
        n = len(vision_objects)
        dists = np.zeros((m, n))

        for i in range(m):
            for j in range(n):
                tid = vision_objects[j].tid
                rid = radar_objects[i].rid
                if map_type == "J3":
                    map_rid = self.mapping_J3[tid]
                elif map_type == "ars408":
                    map_rid = self.mapping_reVeh[tid]
                else:
                    map_rid = self.mapping[tid]
                rv_x_dist = abs(radar_objects[i].x - vision_objects[j].x)
                rv_y_dist = abs(radar_objects[i].y - vision_objects[j].y)
                # 加权欧几里得距离，应用权重并计算距离，y方向权重3，x方向权重1
                weighted_distance = math.sqrt(rv_x_dist ** 2 + (3 * rv_y_dist) ** 2)
                # 检查雷达目标和视觉目标的 id 是否一致以及它们在 x、y 方向上的距离是否满足条件，来决定是否将它们匹配在一起。这有助于减少误匹配的可能性
                dists[i][j] = weighted_distance if (rid in map_rid) else float('inf')
                # dists[i][j] = rv_x_dist if (rv_y_dist < 4) else float('inf')   # 去除目标一致性匹配
                # if rv_x_dist > 20 or rv_y_dist > max(3, vision_objects[j].x * 0.1)
                vision_distance = math.sqrt(vision_objects[j].x ** 2 + (3 * vision_objects[j].y) ** 2)
                if weighted_distance > max(4, vision_distance * 0.20) or rv_y_dist > max(4, vision_objects[j].y * 0.20):
                    dists[i][j] = float('inf')

        return dists

    def greedy_match(self, dists, radar_objects, vision_objects):
        col = -np.ones(len(vision_objects), dtype=np.int_)
        while dists.shape[0] and dists.shape[1]:
            min_index = np.argmin(dists)
            i, j = np.unravel_index(min_index, dists.shape)
            if dists[i][j] < float('inf'):
                tid = vision_objects[j].tid
                radar_objects[i].tid = tid
                col[j] = i
                dists[i, :] = float('inf')
                dists[:, j] = float('inf')
            else:
                break

        for j, i in enumerate(col):
            if i == -1:
                self.update(vision_objects[j].tid, None, vision_objects[j])
            else:
                self.update(vision_objects[j].tid, radar_objects[i], vision_objects[j])

    def greedy_match_map(self, dists, dists2, dists3, radar_objects, vision_objects, J3_objects, reVeh_objects):
        # 贪心匹配
        col = -np.ones(len(vision_objects), dtype=np.int_)
        while dists.shape[0] and dists.shape[1]:
            min_index = np.argmin(dists)
            i, j = np.unravel_index(min_index, dists.shape)
            if dists[i][j] < float('inf'):
                tid = vision_objects[j].tid
                radar_objects[i].tid = tid
                col[j] = i
                dists[i, :] = float('inf')
                dists[:, j] = float('inf')
            else:
                break

        col2 = -np.ones(len(vision_objects), dtype=np.int_)
        while dists2.shape[0] and dists2.shape[1]:
            min_index = np.argmin(dists2)
            i, j = np.unravel_index(min_index, dists2.shape)
            if dists2[i][j] < float('inf'):
                tid = vision_objects[j].tid
                J3_objects[i].tid = tid
                col2[j] = i
                dists2[i, :] = float('inf')
                dists2[:, j] = float('inf')
            else:
                break

        col3 = -np.ones(len(vision_objects), dtype=np.int_)
        while dists3.shape[0] and dists3.shape[1]:
            min_index = np.argmin(dists3)
            i, j = np.unravel_index(min_index, dists3.shape)
            if dists3[i][j] < float('inf'):
                tid = vision_objects[j].tid
                reVeh_objects[i].tid = tid
                col3[j] = i
                dists3[i, :] = float('inf')
                dists3[:, j] = float('inf')
            else:
                break

        for i in range(len(vision_objects)):
            radar_index = col[i]
            J3_index = col2[i]
            reVeh_index = col3[i]
            radar_object = radar_objects[radar_index] if radar_index > -1 else None
            J3_object = J3_objects[J3_index] if J3_index > -1 else None
            # self.update(vision_objects[i].tid, radar_object, vision_objects[i], J3_object)

            reVeh_object = reVeh_objects[reVeh_index] if reVeh_index > -1 else None
            self.update(vision_objects[i].tid, radar_object, vision_objects[i], J3_object, reVeh_object)

    def save_track(self, result_pd, save_alone):
        result_new = []
        for tid in self.tracks.keys():
            result_e = []
            track = self.tracks[tid]
            save_data = track.save_data
            for item in save_data:
                if item.get('rid'):     # 去除未匹配
                    result_e.append(item)
            threshold = 100
            if len(result_e) > 0:
                df_num = pd.DataFrame(result_e)
                # 统计 'class' 为 2，4 的次数
                count_class_2 = (df_num['class'] == 2).sum()
                count_class_4 = (df_num['class'] == 4).sum()
                if count_class_2 > 50 or count_class_4 > 50:
                    threshold = 50

            if len(result_e) >= threshold:      # 把class是2、4的有效帧数阈值改成50 其他类别保留有效帧100以上
                result_new.extend(result_e)

        result_pd.extend(result_new)
        # if save_alone:
        #     df_e = pd.DataFrame(result_new)
        #     df_e.to_csv(os.path.join(self.path, self.filename.replace('.txt', '.csv')), index=False)

    def get_match_rids(self, rids):
        rids = np.array(rids)
        # 计算每个元素的出现次数
        counts = Counter(rids)
        # 获取总元素数量
        total_elements = len(rids)
        # 定义阈值：次数占比超过 1/4 或者出现次数达到至少 30 次
        threshold_count = 30
        threshold_ratio = total_elements / 4
        # 找出满足条件的元素
        frequent_rids = [
            rid for rid, count in counts.items()
            if count > threshold_ratio or count >= threshold_count]

        return frequent_rids

    def track(self, result_pd, save_alone=False):
        for index, pts_e in enumerate(self.pts_data):
            if index >= len(self.vision_data):
                continue
            self.predict()
            try:
                vision_objects = self.load_vision_objects(index)
                radar_objects = self.load_radar_objects(index)
                J3_objects = self.load_J3_objects(index)
                reVeh_objects = self.load_reVeh_objects(index)
            except:
                print(f'代码342行--捕捉到{self.filename} 文件 {index}帧报错')
                continue
            try:
                dists = self.cal_dists(radar_objects, vision_objects)
                dists2 = self.cal_dists(J3_objects, vision_objects)
                dists3 = self.cal_dists(reVeh_objects, vision_objects)
            except:
                print(f'代码348行--捕捉到{self.filename} 文件 {index}帧报错')
                continue

            self.greedy_match_map(dists, dists2, dists3, radar_objects, vision_objects, J3_objects, reVeh_objects)

        # 根据单帧最优匹配的结果（例如贪心匹配），在多次匹配中，统计每个视觉目标对应的雷达目标 id 的出现次数，选择出现次数最多的雷达目标 id 作为该视觉目标的映射关系
        for tid, track in self.tracks.items():
            rids = np.array([o.rid for o in track.radar_data if o])
            # if len(rids) and tid == 18:
            if len(rids):
                # rid = np.argmax(np.bincount(rids))
                # self.mapping[tid] = rid
                self.mapping[tid] = self.get_match_rids(rids)
            else:
                # self.mapping[tid] = -1
                self.mapping[tid] = []
            rids2 = np.array([o.rid for o in track.J3_data if o])
            # if len(rids) and tid == 18:
            if len(rids2):
                # rid = np.argmax(np.bincount(rids))
                # self.mapping[tid] = rid
                self.mapping_J3[tid] = self.get_match_rids(rids2)
            else:
                # self.mapping[tid] = -1
                self.mapping_J3[tid] = []
            rids3 = np.array([o.rid for o in track.reVeh_data if o])
            # if len(rids) and tid == 18:
            if len(rids3):
                # rid = np.argmax(np.bincount(rids))
                # self.mapping[tid] = rid
                self.mapping_reVeh[tid] = self.get_match_rids(rids3)
            else:
                # self.mapping[tid] = -1
                self.mapping_reVeh[tid] = []
        pass
        # 清空跟踪结果
        self.tracks = dict()
        for index, pts_e in enumerate(self.pts_data):
            if index >= len(self.vision_data):
                continue
            self.predict()
            try:
                vision_objects = self.load_vision_objects(index)
                radar_objects = self.load_radar_objects(index)
                J3_objects = self.load_J3_objects(index)
                reVeh_objects = self.load_reVeh_objects(index)
            except:
                continue

            try:
                dists = self.cal_dists_mapping(radar_objects, vision_objects)
                dists2 = self.cal_dists_mapping(J3_objects, vision_objects, map_type="J3")
                dists3 = self.cal_dists_mapping(reVeh_objects, vision_objects, map_type="ars408")
            except:

                continue
            self.greedy_match_map(dists, dists2, dists3, radar_objects, vision_objects, J3_objects, reVeh_objects)

        self.save_track(result_pd, save_alone)

    def load_radar_objects(self, img_index):
        radar_objects = []
        if self.radar_data.get(img_index, None) == None or self.Signal_data.get(img_index, None) == None:
            return radar_objects
        data_dict = self.radar_data.get(img_index, {})
        Signal_index_data = self.Signal_data.get(img_index, {})
        if len(Signal_index_data) == 0:
            return radar_objects
        elif 'vehCan' in Signal_index_data:
            Signal_f32Speed_mean = np.mean([current_vehCan['f32Speed'] for current_vehCan in Signal_index_data["vehCan"]]) * (1 / 3.6)  # Signal数据f32Speed单位是km/h , * (1 / 3.6) 转为m/s
        else:
            Signal_f32Speed_mean = np.mean([current_vehCan['speed'] for current_vehCan in Signal_index_data["carSignal"]]) * (1 / 3.6)  # Signal数据f32Speed单位是km/h , * (1 / 3.6) 转为m/s

        if len(self.radar_data.get(img_index, {})) == 0:
            return radar_objects

        for data_e in data_dict:
            rid = data_e.get('id', 999)
            x = data_e.get('pos_lon', 999) if data_e.get('pos_lon') is not None else 999
            prob = data_e.get('ExstProb', 999) if data_e.get('ExstProb') is not None else 999
            if 'VabsYStd' in data_e:
                y = data_e.get('pos_lat', 999) if data_e.get('pos_lat') is not None else 999
                vx = data_e.get('VabsX', 999) if data_e.get('VabsX') is not None else 999
                vy = data_e.get('VabsY', 999) if data_e.get('VabsY') is not None else 999
                heading = data_e.get('Orientation', 999) if data_e.get('Orientation') is not None else 999
            else:
                # 老雷达
                y = -data_e.get('pos_lat', 999) if data_e.get('pos_lat') is not None else 999
                vx = data_e.get('vel_lon', 999) if data_e.get('vel_lon') is not None else 999
                vy = data_e.get('vel_lat', 999) if data_e.get('vel_lat') is not None else 999
                heading = data_e.get('angle', 999)
                if vx != 999:
                    vx = vx + Signal_f32Speed_mean

            radar_objects.append(RadarObject(x, y, prob, vx, vy, rid))
            save_data = {'rid': rid, 'frame_index': img_index, 'r_x':x, 'r_y':y, 'r_vx': vx, 'r_vy':vy, 'r_heading':heading}
            radar_objects[-1].save_data = save_data


        if len(self.corner_radar_data.get(img_index, {})) == 0:
            return radar_objects
        data_dict = self.corner_radar_data.get(img_index, {})

        for data_e in data_dict:
            x = data_e.get('longDistance', 999)
            y = -data_e.get('latDistance', 999)
            prob = data_e.get('confidence', 999) / 100
            vx = data_e.get('vel_lon', 999)  # 角雷达是相对速度
            vy = data_e.get('vel_lat', 999)  # 角雷达是相对速度
            if vx != 999:
                vx = vx + Signal_f32Speed_mean
            heading = data_e.get('heading', 999)
            rid = data_e.get('id', 999)
            radar_objects.append(RadarObject(x, y, prob, vx, vy, rid))
            save_data = {'rid': rid, 'frame_index': img_index, 'r_x':x, 'r_y':y, 'r_vx': vx, 'r_vy':vy, 'r_heading':heading}
            radar_objects[-1].save_data = save_data

        return radar_objects

    def load_vision_objects(self, img_index):
        vision_objects = []
        if self.vision_data.get(img_index, None) == None:
            return vision_objects
        if self.vision_keyword in ['_arcsoft_obj.txt'] or self.vision_keyword == '.txt':
            if self.mode:
                data_dict = self.vision_data.get(img_index, {}) 
            else:
                visInfo = self.vision_data.get(img_index, {}).get("Oem", {}).get("visInfo", [])
                if len(visInfo) > 0:
                    data_dict = visInfo[0].get("obj", {}).get("objAttributes", {})
                else:
                    return vision_objects
            # data_dict = self.vision_data.get(img_index, {}) if self.mode else {'objAttributes': self.vision_data.get(img_index, {})["perception"]["objects"]}  # pc
            # Signal_img_index_data = self.Signal_data.get(img_index, {})[str(img_index)]
            # Signal_u64ImagePtsTime = Signal_img_index_data["timeSync"]["u64ImagePtsTime"]
            # Signal_f32Speed_mean = np.mean([current_vehCan['f32Speed'] for current_vehCan in Signal_img_index_data["vehCan"]]) * (1 / 3.6)  # Signal数据f32Speed单位是km/h , * (1 / 3.6) 转为m/s

            # Signal_u64ImagePtsTime = Signal_img_index_data["frameInfo"]["u64BootTimestamp"]
            # Signal_f32Speed_mean = np.mean(
            #     [current_vehCan['speed'] for current_vehCan in Signal_img_index_data["carSignal"]]) * (
            #                            1 / 3.6)  # Signal数据f32Speed单位是km/h , * (1 / 3.6) 转为m/s

            for data_e in data_dict:
                ObjClass = data_e.get('u8OemObjClass')
                if ObjClass in [0, 1, 2, 3, 4, 6, 7] and data_e.get('f32LatDistance'):   # 临时措施防止算法结果为null
                    x = data_e.get('f32LongDistance', 999)
                    y = data_e.get('f32LatDistance', 999)
                    l = data_e.get('f32Length', 999)
                    w = data_e.get('f32Width', 999)
                    h = data_e.get('f32Height', 999)

                    angle = data_e.get('f32AngleRate', 999)
                    tid = data_e.get('u8Id', 999)
                    conf = data_e.get('ExPro', 1)
                    # vx = data_e.get('f32RelativeLongVelocity', 999)
                    # vy = data_e.get('f32RelativeLatVelocity', 999)
                    vx = data_e.get('f32AbsoluteLongVelocity', 999)
                    vy = data_e.get('f32AbsoluteLatVelocity', 999)
                    cipv = data_e.get('s32Cipv', 999)
                    heading = data_e.get('f32Heading', 999)
                    # # 标志位为1的帧的横向KPI
                    # use_ratio_y = data_e.get('use_ratio_y')
                    age = data_e.get('u16ObjectAge', 999)
                    vision_objects.append(VisionObject(tid, conf, w, h, x, y, vx, vy, angle, img_index, 'type'))
                    save_data = {'filename':self.filename, 'CIPV':cipv, 'tid':tid, 'age':age, 'frame_index': img_index, 'x': x, 'y':y, 'class': ObjClass, 'vx':vx, 'vy':vy,'heading':heading}  # , 'Signal_Time':Signal_u64ImagePtsTime, 'Signal_Speed':Signal_f32Speed_mean
                    vision_objects[-1].save_data = save_data

        elif self.vision_keyword in ['_J3_obj.txt']:
            data_dict = self.vision_data.get(img_index, {})
 
            for data_e in data_dict:
                ObjClass = data_e.get('type')
                if ObjClass in [1, 2, 3]:   #  0未知 1四轮车 2二轮车 3 行人 4交通物，屏蔽未知和交通物
                    x = data_e.get('obstacle_pos_x', 999)
                    y = data_e.get('obstacle_pos_y', 999)
                    l = data_e.get('obstacle_length', 999)
                    w = data_e.get('obstacle_width', 999)
                    h = data_e.get('obstacle_height', 999)
                    angle = data_e.get('obstacle_angle', 999)
                    tid = data_e.get('id', 999)
                    conf = data_e.get('conf', 1)
                    vx = data_e.get('obstacle_vel_x', 999)
                    vy = data_e.get('obstacle_vel_y', 999)
                    cipv = data_e.get('cipv_flag', 999)
                    age = data_e.get('age', 999)
                    vision_objects.append(VisionObject(tid, conf, w, h, x, y, vx, vy, angle, img_index, type))
                    save_data = {'filename': self.filename, 'CIPV': cipv, 'tid': tid, 'age':age, 'frame_index': img_index, 'x': x, 'y':y, 'class': ObjClass, 'vx':vx, 'vy':vy}
                    vision_objects[-1].save_data = save_data

        return vision_objects

    def load_J3_objects(self, img_index):
        J3_objects = []
        if self.J3_data.get(img_index, None) == None:
            return J3_objects
        if self.J3_data != []:
            data = self.J3_data.get(img_index, {})
            for data_e in data:
                ObjClass = data_e.get('type')
                if ObjClass in [1, 2, 3]:   #  0未知 1四轮车 2二轮车 3 行人 4交通物，屏蔽未知和交通物
                    x = data_e.get('obstacle_pos_x', 999)
                    y = data_e.get('obstacle_pos_y', 999)
                    l = data_e.get('obstacle_length', 999)
                    w = data_e.get('obstacle_width', 999)
                    h = data_e.get('obstacle_height', 999)
                    angle = data_e.get('obstacle_angle', 999)
                    tid = data_e.get('id', 999)
                    conf = data_e.get('conf', 1)
                    vx = data_e.get('obstacle_vel_x', 999)
                    vy = data_e.get('obstacle_vel_y', 999)
                    cipv = data_e.get('cipv_flag', 999)
                    age = data_e.get('age', 999)
                    J3_objects.append(RadarObject(x, y, conf, vx, vy, tid))
                    save_data = {'filename': self.filename, 'J3_CIPV': cipv, 'J3_tid': tid, 'J3_age':age, 'frame_index': img_index, 'J3_x': x, 'J3_y':y, 'J3_class': ObjClass, 'J3_vx':vx, 'J3_vy':vy}
                    J3_objects[-1].save_data = save_data

        return J3_objects

    def load_reVeh_objects(self, img_index):
        vision_objects = []
        if self.arc_reVeh_data.get(img_index, None) == None:
            return vision_objects
        
        data_dict = self.arc_reVeh_data.get(img_index, {})
        for data_e in data_dict:
            ObjClass = data_e.get('u8OemObjClass')
            if ObjClass in [0, 1, 2, 3, 4, 6, 7] and data_e.get('f32LatDistance'):   # 临时措施防止算法结果为null
                x = data_e.get('f32LongDistance', 999)
                y = data_e.get('f32LatDistance', 999)
                l = data_e.get('f32Length', 999)
                w = data_e.get('f32Width', 999)
                h = data_e.get('f32Height', 999)

                angle = data_e.get('f32AngleRate', 999)
                tid = data_e.get('u8Id', 999)
                conf = data_e.get('ExPro', 1)
                # vx = data_e.get('f32RelativeLongVelocity', 999)
                # vy = data_e.get('f32RelativeLatVelocity', 999)
                vx = data_e.get('f32AbsoluteLongVelocity', 999)
                vy = data_e.get('f32AbsoluteLatVelocity', 999)
                cipv = data_e.get('s32Cipv', 999)
                heading = data_e.get('f32Heading', 999)
                # # 标志位为1的帧的横向KPI
                # use_ratio_y = data_e.get('use_ratio_y')
                age = data_e.get('u16ObjectAge', 999)
                vision_objects.append(RadarObject(x, y, conf, vx, vy, tid))
                save_data = {'filename':self.filename, 'reVeh_CIPV':cipv, 'reVeh_tid':tid, 'reVeh_age':age, 'reVeh_frame_index': img_index, 'reVeh_x': x, 'reVeh_y':y, 'reVeh_class': ObjClass, 'reVeh_vx':vx, 'reVeh_vy':vy,'reVeh_heading':heading}  # , 'Signal_Time':Signal_u64ImagePtsTime, 'Signal_Speed':Signal_f32Speed_mean
                vision_objects[-1].save_data = save_data

        return vision_objects
    
def save_to_csv(data, output_path):
    df = pd.DataFrame(data)
    # if all(col in df.columns for col in ['vx','Signal_Speed','r_vx']):
    #     df['abs_arc_vx'] = df['vx'] + df['Signal_Speed']
    #     df['abs_r_vx'] = df['r_vx'] + df['Signal_Speed']
    #     df[f'diff_abs_arc_vx'] = df['abs_arc_vx'].diff()
    #     df[f'diff_abs_r_vx'] = df['abs_r_vx'].diff()
    df.to_csv(output_path, index=False)

# def process_file(args):
#     # """单个文件的处理函数"""
#     # vision_file, vision_keyword, mode, map_relate = args
#     # logger.info(f'{vision_file} 目标自动匹配中...')
#     # model = RadarVisionHybrid(vision_file, vision_keyword, mode, map_relate)
#     # result_pd = []
#     # model.track(result_pd, save_alone=True)
#     # return result_pd
#     vision_file, vision_keyword, mode, map_relate = args
#     logger.info(f'{vision_file} 目标自动匹配中...')

#     # try:
#     model = RadarVisionHybrid(vision_file, vision_keyword, mode, map_relate)
#     result_pd = []
#     model.track(result_pd, save_alone=True)
#     return result_pd
#     # except FileNotFoundError as e:
#     #     logger.warning(f'文件缺失，跳过: {e.filename}')
#     #     return []
#     # except Exception as e:
#     #     logger.error(f'处理 {vision_file} 时出现异常: {str(e)}')
#     #     return []

def process_file(args):
    vision_file, vision_keyword, mode, map_relate = args
    logger.info(f'{vision_file} 目标自动匹配中...')

    try:
        model = RadarVisionHybrid(vision_file, vision_keyword, mode, map_relate)
        result_pd = []
        model.track(result_pd, save_alone=True)
        return result_pd

    except Exception as e:
        import traceback
        logger.error(f'❌ 处理 {vision_file} 时出现异常: {e}')
        logger.error(traceback.format_exc())
        return []

def main_multiple_processes(file_all, vision_keyword, src_path, mode, map_relate, batch_size=1000000):
    # 设置进程池大小，可以根据硬件调整
    num_processes = max(1, mp.cpu_count() - 2)
    # num_processes = 1
    pool = mp.Pool(processes=num_processes)
    now = datetime.now()
    formatted_time = now.strftime('%Y%m%d%H%M')
    try:
        # 并行处理文件列表，生成迭代器.遍历迭代时生成对应结果
        results_iterator = pool.imap(process_file, zip(file_all, [vision_keyword] * len(file_all),[mode] * len(file_all),[map_relate] * len(file_all)))

        buffer = []
        batch_counter = 0  # 记录当前批次号，达到百万量级时保存
        for i, result in enumerate(tqdm(results_iterator, total=len(file_all), desc="任务正在处理..."), start=1):
            buffer.extend(result)
            if len(buffer) >= batch_size:
                batch_counter += 1
                output_filename = os.path.join(src_path, f'{formatted_time}_output_part{batch_counter}.csv')
                save_to_csv(buffer, output_filename)
                logger.info(f'已保存 {len(buffer)} 条数据到 {output_filename}')
                buffer.clear()

        # 处理剩余的结果
        if buffer:
            batch_counter += 1
            output_filename = os.path.join(src_path, f'{formatted_time}_output_part{batch_counter}.csv')
            save_to_csv(buffer, output_filename)
            logger.info(f'已保存 {len(buffer)} 条数据到 {output_filename}')
    finally:
        # 确保关闭进程池
        pool.close()
        pool.join()


def make_parser():
    parser = argparse.ArgumentParser("雷达真值自动匹配")
    parser.add_argument(
        "--all_result_path",
        "--所有结果路径（回灌后的arc算法结果、J3、前雷达、角雷达、车机信号等结果、视频）",
        type=str,
        default=r'',  # D:\1\kpi_person
        help="Input your all_result_path.",
    )
    parser.add_argument(
        "--recharge_result_path",
        "--回灌后的arc算法结果",
        type=str,
        # default=r'\\hz-iotfs02\Model_Test\TestSpace\Personal_Space\CHZ\Southlake\adas_perception_v3.1_SPC030_2m_80\output\shaoyuqi\other\22',
        default=r'\\hz-iotfs02\Model_Test\TestSpace\Personal_Space\CHZ\Southlake\adas_perception_v3.1_SPC030_2m_80\output\shaoyuqi\20260116_V3.1_2M_3.1.27223.1457',
        help="Input your recharge_result_path.",
    )
    parser.add_argument(
        "--virtual_test_set_path",
        "--回灌后的arc算法结果对应视频所在的虚拟测试集路径",
        type=str,
        # default=r'',
        default=r"\\hz-iotfs02\Function_Test\Front_Camera\#24658_FT_ADAS\shaoyuqi\cesucejuV3",
        help="Input your virtual_test_set_path.",
    )

    parser.add_argument(
        "--virtual_test_txt_path",
        "--回灌后的arc算法结果对应视频所在的虚拟测试集路径 txt文本路径",
        type=str,
        default=r'',  # D:\1\test\shengyusucai\DTC\lixiang1.txt  剩余素管路径  F:\project\FT\lixiang_path.txt
        help="Input your virtual_test_txt_path.",
    )
    return parser

def txt_path_dic(path,key_ends='.txt'):
    txt_name_path_dic = {}
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(key_ends) and '_arcsoft_obj' not in file:
                txt_path = os.path.join(root, file)
                txt_name_path_dic[file] = txt_path
    return txt_name_path_dic

if __name__ == '__main__':
    try:
        args = make_parser().parse_args()
        if args.all_result_path and os.path.exists(args.all_result_path):
            mode = True
            src_path = args.all_result_path
            vision_keyword = '_arcsoft_obj.txt'   # _arcsoft_obj.txt   _J3_obj.txt
            map_relate = None  # No map_relate needed in this mode
            # 视频所在路径视频数量大于回灌结果数量，以回灌结果为准对数据加以限制
            # txt_name_path_dic = txt_path_dic(args.recharge_result_path, key_ends='.txt')
            file_all = []
            for root, dirs, files in os.walk(src_path):
                for file in files:
                    if file.endswith(vision_keyword):
                    # 将上方txt_name_path_dic = txt_path_dic注释代码打开
                    # 视频所在路径视频数量大于回灌结果数量，以回灌结果为准对数据加以限制
                    # if file.endswith(vision_keyword) and file.replace(vision_keyword,'.txt') in txt_name_path_dic:

                        vision_file = os.path.join(root, file)
                        file_all.append(vision_file)

        elif args.recharge_result_path and os.path.exists(args.recharge_result_path) and args.virtual_test_set_path and os.path.exists(args.virtual_test_set_path):
            mode = False
            src_path = args.recharge_result_path
            vision_keyword = '.txt'
            txt_name_path_dic = txt_path_dic(args.recharge_result_path, vision_keyword)
            map_relate = {}
            if os.path.isdir(args.virtual_test_set_path):
                for root, dirs, files in os.walk(args.virtual_test_set_path):
                    for file in files:
                        if file.endswith('.h264') and file.replace('.h264', '.txt') in txt_name_path_dic:
                            h264_file = os.path.join(root, file)
                            map_relate[file.replace('.h264', '.txt')] = os.path.join(root, 'log', file.replace('.h264', '.txt'))
            elif os.path.isfile(args.virtual_test_set_path):
                with open(args.virtual_test_set_path, "r", encoding="utf-8", errors="ignore") as f:
                    for idx, line in enumerate(f, start=1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            filename = os.path.basename(line)
                            root = os.path.dirname(line)
                            if filename.endswith('.h264') and filename.replace('.h264', '.txt') in txt_name_path_dic:
                                map_relate[filename.replace('.h264', '.txt')] = os.path.join(root, 'log', filename.replace('.h264', '.txt'))
                        except json.JSONDecodeError:
                            print(f"⚠️ 第{idx}行不是合法 JSON: {line}")

            # file_all = list(txt_name_path_dic.values())
            common_keys = txt_name_path_dic.keys() & map_relate.keys()
            file_all = [txt_name_path_dic[k] for k in common_keys]

        # 回灌结果txt文件，虚拟测试集路径txt文件
        elif args.recharge_result_path and os.path.exists(args.recharge_result_path) and args.virtual_test_txt_path and os.path.exists(args.virtual_test_txt_path):
            mode = False
            src_path = args.recharge_result_path
            vision_keyword = '.txt'
            txt_name_path_dic = txt_path_dic(args.recharge_result_path,vision_keyword)
            map_relate = {}
            with open(args.virtual_test_txt_path, 'r', encoding='utf-8') as txt_pd:
                txt_list = txt_pd.readlines()
                for file in txt_list:
                    file = file.strip()
                    fiel_name = os.path.basename(file)
                    if file.endswith('.h264') and fiel_name.replace('.h264', '.txt') in txt_name_path_dic:
                        map_relate[fiel_name.replace('.h264', '.txt')] = os.path.join(os.path.dirname(file), 'log', fiel_name.replace('.h264', '.txt'))
            # file_alls = list(txt_name_path_dic.values())
            # 统计共有的 文件
            common_keys = txt_name_path_dic.keys() & map_relate.keys()
            file_all = [txt_name_path_dic[k] for k in common_keys]

        else:
            print('程序退出！请检查路径参数是否正确')
            exit()

        main_multiple_processes(file_all, vision_keyword, src_path, mode, map_relate)  # Pass mode and map_relate

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        tb_info = traceback.format_tb(exc_tb)
        logger.error(tb_info)
        logger.error(exc_obj)
