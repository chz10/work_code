# -*- coding: utf-8 -*-
import os
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
import traceback
import sys
import pickle
warnings.filterwarnings('ignore')

class LabelInfoProcessor:
    """标签信息处理类"""
    @staticmethod
    def get_special_project_labelinfo(input_path):
        if os.path.splitext(input_path)[1] == ".xlsx":
            return LabelInfoProcessor._process_excel_file(input_path)
        else:
            raise ValueError(f"不支持的文件格式：{input_path}")
        
    @staticmethod
    def _process_excel_file(input_path):
        """处理Excel文件"""
        labelinfo_df = pd.read_excel(input_path, sheet_name="Sheet1", dtype=str)
        result = {}
        pre_scene = ""
        
        for index, row in labelinfo_df.iterrows():
            case_name = str(row.get("case_name", "")).strip()
            cur_scene = str(row.get("scene", "")).strip()
            
            # 大类继承
            if not cur_scene or cur_scene == "nan":
                cur_scene = pre_scene
            else:
                pre_scene = cur_scene

            filename = str(row.get("文件名", "")).strip()
            video_dir = str(row.get("数据路径", "")).strip()
            start = row.get("开始帧", "")
            end = row.get("结束帧", "")
            target_id = str(row.get("真实压线目标ID", "")).strip()
            class_key = row.get("目标类别", None)
            class_value = row.get("class_value", None)
            
            # 缺少必要字段，跳过
            if not filename or not start or not end:
                continue
            
            # 转为整数
            try:
                start_frame = int(float(start)) - 50
                end_frame = int(float(end)) + 50
            except ValueError:
                continue

            # 聚合逻辑
            if filename not in result:
                result[filename] = []

            try:
                target_id = int(target_id)
            except ValueError:
                pass
                
            result[filename].append({
                "line_num": index + 1,
                "case_name": case_name,
                "scene": cur_scene,
                "filename": filename,
                "video_dir": video_dir,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "target_id_list": target_id,
                "class_key": class_key,
                "class_value": class_value
            })
        
        return result
    
    @staticmethod
    def find_specified_files(input_path, file_extensions=(".h264", ".h265")):
        search_results = {}
        if os.path.isdir(input_path):
            for root, _, files in os.walk(input_path):
                for file in files:
                    if file.lower().endswith(tuple(file_extensions)):
                        filename = os.path.splitext(file)[0]
                        if filename not in search_results:
                            search_results[filename] = os.path.join(root, file)
        elif os.path.isfile(input_path):
            with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
                for idx, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        filename = os.path.splitext(os.path.basename(line))[0]
                        search_results[filename] = line
                    except json.JSONDecodeError:
                        print(f"⚠️ 第{idx}行不是合法 JSON: {line}")
        else:
            print("素材路径不存在")
        return search_results


logger.add("missing_files351.log", encoding="utf-8", rotation="5 MB", retention="10 days", level="ERROR")

def load_txt(path, frame_list):
    values = []
    if not os.path.exists(path):
        logger.error(f"缺失文件: {path}")
        return values
    with open(path, 'r', encoding='utf-8') as txt_pd:
        for index, line in enumerate(txt_pd, start=1):
            try:
                data = json.loads(line.strip())
                if len(data) == 0:
                    continue
                # 提取帧号
                frame = None
                if "Oem" in data:
                    frame = data["Oem"]["u64FrameId"]
                else:
                    # 尝试从字典键中获取帧号
                    for key in data.keys():
                        if key.isdigit():
                            frame = int(key)
                            break
                if frame is None or frame not in frame_list:
                    continue
                values.append(data)
            except json.JSONDecodeError:
                print(f"*********无法解析  {path} 文件的第{index}行")
    return values


def is_valid(value):
    """检查值是否有效"""
    return value is not None and value != '' and value != 'null' and value != 'undefined'


def extract_lane_data_arc(file_path):
    """从arcsoft结果文件中提取车道线数据"""
    lane_data_dict = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='gbk', errors='ignore') as file:
            lines = file.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            
            if isinstance(data, dict):
                if "Oem" in data:
                    frame = data["Oem"]["u64FrameId"]
                    vis_info = data["Oem"].get("visInfo", [])
                    for item in vis_info:
                        lane_attributes = item.get("lane", {}).get("laneAttributes", [])
                        for attr in lane_attributes:
                            lane_id = attr.get("u8Id", None)
                            C0_values = attr.get("f32LineC0", None)
                            C1_values = attr.get("f32LineC1", None)
                            C2_values = attr.get("f32LineC2", None)
                            C3_values = attr.get("f32LineC3", None)
                            u8OemSide = attr.get("u8OemSide", None)
                            u8OemType = attr.get("u8OemType", None)
                            u8Age = attr.get("u8Age", None)
                            u8OemColor = attr.get("u8OemColor", None)
                            f32Probability = attr.get("f32Probability", None)
                            f32StartX = attr.get("f32StartX", None)
                            f32StartY = attr.get("f32StartY", None)
                            f32EndX = attr.get("f32EndX", None)
                            f32EndY = attr.get("f32EndY", None)
                            s32LaneTypeClass = attr.get("s32LaneTypeClass", None)
                            s32LaneLocation = attr.get("s32LaneLocation", None)

                            if frame not in lane_data_dict:
                                lane_data_dict[frame] = {}

                            lane_data_dict[frame][lane_id] = {
                                "u8Id": lane_id if is_valid(lane_id) else None,
                                "f32LineC0": C0_values if is_valid(C0_values) else None,
                                "f32LineC1": C1_values if is_valid(C1_values) else None,
                                "f32LineC2": C2_values if is_valid(C2_values) else None,
                                "f32LineC3": C3_values if is_valid(C3_values) else None,
                                "u8OemSide": u8OemSide if is_valid(u8OemSide) else None,
                                "u8OemType": u8OemType if is_valid(u8OemType) else None,
                                "u8Age": u8Age if is_valid(u8Age) else None,
                                "u8OemColor": u8OemColor if is_valid(u8OemColor) else None,
                                "f32Probability": f32Probability if is_valid(f32Probability) else None,
                                "f32StartX": f32StartX if is_valid(f32StartX) else None,
                                "f32StartY": f32StartY if is_valid(f32StartY) else None,
                                "f32EndX": f32EndX if is_valid(f32EndX) else None,
                                "f32EndY": f32EndY if is_valid(f32EndY) else None,
                                "s32LaneTypeClass": s32LaneTypeClass if is_valid(s32LaneTypeClass) else None,
                                "s32LaneLocation": s32LaneLocation if is_valid(s32LaneLocation) else None
                            }
        except json.JSONDecodeError as e:
            print(f"解析JSON错误: {e}, 行内容: {line[:100]}...")
            continue
    return lane_data_dict


class Lanetouchline_flagChecker:
    """目标压线检测器"""
    def __init__(self, lane_point_density=0.2, cache_dir=None):
        """
        初始化压线检测器
        Args:
            lane_point_density: 车道线离散点密度（米）
            cache_dir: 缓存目录，用于存储解析的文件数据
        """
        self.lane_point_density = lane_point_density
        self.cache_dir = cache_dir
        self._objdata_cache = {}
        self._lanedata_cache = {}
        self._model_cache = {}
        
        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
    
    def _get_cache_key(self, filepath, data_type):
        """获取缓存键"""
        return f"{filepath}_{data_type}"
    
    def _load_from_cache(self, filepath, data_type):
        """从缓存加载数据"""
        if not self.cache_dir:
            return None
            
        cache_key = self._get_cache_key(filepath, data_type)
        cache_file = os.path.join(self.cache_dir, f"{hash(cache_key)}.pkl")
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except:
                pass
        return None
    
    def _save_to_cache(self, filepath, data_type, data):
        """保存数据到缓存"""
        if not self.cache_dir:
            return
            
        cache_key = self._get_cache_key(filepath, data_type)
        cache_file = os.path.join(self.cache_dir, f"{hash(cache_key)}.pkl")
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
        except:
            pass
    
    def calculate_object_corners(self, obj):
        """
        计算目标框的四个角点（世界坐标系）
        """
        x_long = obj.get("f32LongDistance")
        y_lat = obj.get("f32LatDistance")
        length = obj.get("f32Length")
        width_ = obj.get("f32Width")
        heading_rad = obj.get("f32Heading")
        
        if x_long is None or y_lat is None or length is None or width_ is None or heading_rad is None:
            return None
        
        # 局部坐标：绕底边中点旋转
        half_w = width_ / 2.0
        corners_local = np.array([
            [length,  half_w],   # 前左
            [length, -half_w],   # 前右
            [0,      -half_w],   # 后右
            [0,       half_w]    # 后左
        ])
        
        # 旋转矩阵
        cos_h = np.cos(heading_rad)
        sin_h = np.sin(heading_rad)
        R = np.array([[cos_h, -sin_h],
                      [sin_h,  cos_h]])
        
        # 旋转并平移到世界坐标系
        corners_world = corners_local @ R.T + np.array([x_long, y_lat])
        
        return corners_world
    
    def generate_lane_points(self, lane, start_x=None, end_x=None):
        """
        生成车道线的离散点
        """
        if end_x is None:
            end_x = 200  # 默认到200米
        if start_x is None:
            start_x = lane.get("f32StartX")
            
        c0 = lane.get("f32LineC0")
        c1 = lane.get("f32LineC1")
        c2 = lane.get("f32LineC2")
        c3 = lane.get("f32LineC3")
        
        if c0 is None or c1 is None or c2 is None or c3 is None:
            return np.array([])
        
        # 根据密度生成点
        dx = self.lane_point_density
        num_points = max(int(abs(end_x - start_x) / dx) + 1, 2)
        xs = np.linspace(start_x, end_x, num_points)
        ys = c0 + c1 * xs + c2 * xs**2 + c3 * xs**3
        
        return np.column_stack((xs, ys))
    
    def check_object_touchline_flag_with_lane(self, obj, lane, lane_width=0.10):
        """
        检查目标是否与车道线碰撞/压线
        """
        touchline_flag, obj_poly, lane_polys, hit_idx = self.check_boundary_touchline_flag(obj, lane, lane_width)
        
        # 计算详细距离信息
        obj_x = obj.get("f32LongDistance")
        obj_y = obj.get("f32LatDistance")
        lane_y_at_obj_x = self._get_lane_y_at_x(lane, obj_x)
        
        distance_to_lane = abs(obj_y - lane_y_at_obj_x) if obj_y is not None and lane_y_at_obj_x is not None else float('inf')
        
        return {
            "touchline_flag": touchline_flag,
            "distance_to_lane": distance_to_lane,
            "object_y": obj_y,
            "lane_y_at_object_x": lane_y_at_obj_x,
            "lane_id": lane.get("u8Id"),
            "object_id": obj.get("u8Id"),
            "lane_type": lane.get("u8Id"),
        }

    def _get_axes(self, polygon: np.ndarray):
        """获取多边形的所有分离轴（边的法向）"""
        axes = []
        for i in range(len(polygon)):
            p1 = polygon[i]
            p2 = polygon[(i + 1) % len(polygon)]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-6:
                axes.append(normal / norm)
        return axes

    def _project_polygon(self, polygon: np.ndarray, axis: np.ndarray):
        """将多边形投影到轴上"""
        projections = polygon @ axis
        return projections.min(), projections.max()

    def _sat_overlap(self, poly1: np.ndarray, poly2: np.ndarray) -> bool:
        """SAT 判定：两个凸多边形是否相交"""
        axes = self._get_axes(poly1) + self._get_axes(poly2)
        for axis in axes:
            min1, max1 = self._project_polygon(poly1, axis)
            min2, max2 = self._project_polygon(poly2, axis)
            if max1 < min2 or max2 < min1:
                return False  # 存在分离轴
        return True
    
    def _lane_segment_to_polygon(self, p1, p2, lane_width):
        """将一段车道线中心线，扩展成一个带宽矩形（4 点）"""
        direction = p2 - p1
        length = np.linalg.norm(direction)
        if length < 1e-6:
            return None

        direction = direction / length
        normal = np.array([-direction[1], direction[0]])

        offset = normal * (lane_width / 2)

        return np.array([
            p1 + offset,
            p2 + offset,
            p2 - offset,
            p1 - offset
        ])

    def check_boundary_touchline_flag(self, obj, lane, lane_width=0.10):
        """SAT + 几何体输出（用于 BEV 可视化）"""
        obj_corners = self.calculate_object_corners(obj)
        if obj_corners is None:
            return False, None, [], None
        
        dist = obj.get("f32LongDistance", 0)
        f32Length = obj.get("f32Length", 10)  # 默认10米
        
        lane_points = self.generate_lane_points(lane, max(0, dist - f32Length), dist + f32Length)
        if len(lane_points) == 0:
            return False, obj_corners, [], None

        lane_polys = []
        for i in range(len(lane_points) - 1):
            p1 = lane_points[i]
            p2 = lane_points[i + 1]
            poly = self._lane_segment_to_polygon(
                np.array(p1), np.array(p2), lane_width
            )
            if poly is not None:
                lane_polys.append(poly)

        # SAT 判断
        for idx, poly in enumerate(lane_polys):
            if self._sat_overlap(obj_corners, poly):
                return True, obj_corners, lane_polys, idx

        return False, obj_corners, lane_polys, None

    def _get_lane_y_at_x(self, lane, x):
        """获取车道线在指定x坐标处的y值"""
        c0 = lane.get("f32LineC0")
        c1 = lane.get("f32LineC1")
        c2 = lane.get("f32LineC2")
        c3 = lane.get("f32LineC3")
        
        if c0 is None or c1 is None or c2 is None or c3 is None:
            return None
        
        return c0 + c1 * x + c2 * x**2 + c3 * x**3
    
    def check_all_objects_touchline_flag(self, obj_dict, lane_list, lane_width=0.10):
        """
        检查所有目标与所有车道线的碰撞情况
        """
        results = []
        if not lane_list:
            return results

        obj_id = obj_dict.get("u8Id")
        obj_x = obj_dict.get("f32LongDistance")
        obj_y = obj_dict.get("f32LatDistance")
        
        if obj_x is None or obj_y is None:
            return results
        
        min_distance = float('inf')
        closest_lane_id = None
        closest_lane_type = None
        touchline_flag_detected = False
        
        for lane_id, lane_dict in lane_list.items():
            location = lane_dict.get("s32LaneLocation", None)

            if location not in [1, 2]:  # 只判断主车道线
                continue
            
            lane_y = self._get_lane_y_at_x(lane_dict, obj_x)
            if lane_y is None:
                continue
                
            distance = abs(obj_y - lane_y)
            
            # 更新最小距离
            if distance < min_distance:
                min_distance = distance
                closest_lane_id = lane_dict.get("u8Id")
                closest_lane_type = lane_dict.get("u8Id", 0)
            
            # 检查碰撞
            if not touchline_flag_detected:  # 如果已经检测到碰撞，跳过后续检测
                result = self.check_object_touchline_flag_with_lane(obj_dict, lane_dict, lane_width)
                if result["touchline_flag"]:
                    touchline_flag_detected = True
        
        results.append({
            "object_id": obj_id,
            "touchline_flag": touchline_flag_detected,
            "closest_lane_id": closest_lane_id,
            "closest_lane_type": closest_lane_type,
            "distance_to_lane": min_distance if min_distance != float('inf') else None,
            "object_class": obj_dict.get("u8OemObjClass", 0),
        })
        
        return results


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


class ReplayObject(object):
    def __init__(self, x, y, prob, vx, vy, rid):
        self.x = x
        self.y = y
        self.prob = prob
        self.vx = vx
        self.vy = vy
        self.tid = None
        self.save_data = None
        self.rid = rid


class Track(object):
    def __init__(self, tid, vision_object, replay_object, save_data):
        self.tid = tid
        self.vision_data = [vision_object]
        self.replay_data = [replay_object]
        self.x_model = CVKF1()
        self.y_model = CVKF1()
        self.vx_model = CVKF1()
        self.save_data = [save_data]


class VisionReplayHybrid(object):
    def __init__(self, vision_file, vision_keyword='_arcsoft_obj.txt', mode=True, map_relate=None, frame_list=None):
        self.vision_keyword = vision_keyword
        self.mode = mode  # 将 mode 存储为实例变量
        self.map_relate = map_relate  # 将 map_relate 存储为实例变量
        self.tracks = dict()
        self.frame_list = frame_list
        self.vision_data = load_txt(vision_file, frame_list)
        # 只加载回灌结果，移除雷达相关数据
        self.replay_data = load_txt(vision_file.replace(vision_keyword, '_arcsoft_obj.txt'), frame_list) if mode else load_txt(map_relate[os.path.basename(vision_file)].replace(vision_keyword, '_arcsoft_obj.txt'), frame_list)
        
        # 加载车道线数据
        self.lane_data = extract_lane_data_arc(vision_file)
        
        # 初始化压线检测器
        self.touchline_flag_checker = Lanetouchline_flagChecker(lane_point_density=0.2)
        
        self.path = os.path.dirname(vision_file)
        self.filename = os.path.basename(vision_file)
        self.mapping = dict()

    def update(self, tid, vision_object, replay_object):
        if replay_object:
            save_data = {**vision_object.save_data, **replay_object.save_data}
        else:
            save_data = vision_object.save_data

        if tid in self.tracks.keys():
            track = self.tracks[tid]
            track.vision_data.append(vision_object)
            track.replay_data.append(replay_object)
            track.save_data.append(save_data)
        else:
            self.tracks[tid] = Track(tid, vision_object, replay_object, save_data)

        track = self.tracks[tid]
        if replay_object:
            if track.x_model.is_init is False:
                P0 = np.array([[1, 0], [0, 1]])
                x0 = np.array([[replay_object.x], [0]])
                y0 = np.array([[replay_object.y], [0]])
                vx0 = np.array([[replay_object.vx], [0]])
                track.x_model.init(x0, P0)
                track.y_model.init(y0, P0)
                track.vx_model.init(vx0, P0)
            else:
                track.x_model.update(np.array([[replay_object.x]]))
                track.y_model.update(np.array([[replay_object.y]]))
                track.vx_model.update(np.array([[replay_object.vx]]))

    def predict(self):
        for tid in self.tracks.keys():
            track = self.tracks[tid]
            if track.x_model.is_init:
                track.x_model.predict()
                track.y_model.predict()
                track.vx_model.predict()

    def cal_dists(self, replay_objects, vision_objects):
        m = len(replay_objects)
        n = len(vision_objects)
        dists = np.zeros((m, n))

        for i in range(m):
            for j in range(n):
                tid = vision_objects[j].tid
                track = self.tracks[tid] if tid in self.tracks.keys() else None

                rv_x_dist = abs(replay_objects[i].x - vision_objects[j].x)
                rv_y_dist = abs(replay_objects[i].y - vision_objects[j].y)

                conf = getattr(replay_objects[i], "prob", 1.0)
                r_prob_dist = (1 - conf)
                rv_vx_dist = abs(replay_objects[i].vx - vision_objects[j].vx)
                r_x_dist = r_y_dist = 0
                r_vx_dist = 0
                
                if track is not None and track.x_model.is_init:
                    pred_x = track.x_model.x[0]
                    pred_y = track.y_model.x[0]
                    pred_vx = track.vx_model.x[0]
                    r_x_dist = abs(replay_objects[i].x - pred_x)
                    r_y_dist = abs(replay_objects[i].y - pred_y)
                    r_vx_dist = abs(replay_objects[i].vx - pred_vx)

                # 提高y轴权重为x轴3倍
                dists[i][j] = 2 * rv_x_dist + 2 * rv_y_dist * 3 + 5 * r_prob_dist + \
                              rv_vx_dist + r_x_dist + r_y_dist + r_vx_dist
                
                gate_x = max(3, vision_objects[j].x * 0.1)
                gate_y = max(3, vision_objects[j].y * 0.1)
                
                if rv_x_dist > gate_x or rv_y_dist > gate_y:
                    dists[i][j] = float('inf')

        return dists

    def cal_dists_mapping(self, replay_objects, vision_objects):
        m = len(replay_objects)
        n = len(vision_objects)
        dists = np.zeros((m, n))

        for i in range(m):
            for j in range(n):
                tid = vision_objects[j].tid
                rid = replay_objects[i].rid
                map_rid = self.mapping[tid]
                
                rv_x_dist = abs(replay_objects[i].x - vision_objects[j].x)
                rv_y_dist = abs(replay_objects[i].y - vision_objects[j].y)
                
                weighted_distance = math.sqrt(rv_x_dist ** 2 + (3 * rv_y_dist) ** 2)
                dists[i][j] = weighted_distance if (rid in map_rid) else float('inf')
                
                vision_distance = math.sqrt(vision_objects[j].x ** 2 + (3 * vision_objects[j].y) ** 2)
                if weighted_distance > max(4, vision_distance * 0.20) or rv_y_dist > max(4, vision_objects[j].y * 0.20):
                    dists[i][j] = float('inf')

        return dists

    def greedy_match(self, dists, replay_objects, vision_objects):
        col = -np.ones(len(vision_objects), dtype=np.int_)
        while dists.shape[0] and dists.shape[1]:
            min_index = np.argmin(dists)
            i, j = np.unravel_index(min_index, dists.shape)
            if dists[i][j] < float('inf'):
                tid = vision_objects[j].tid
                replay_objects[i].tid = tid
                col[j] = i
                dists[i, :] = float('inf')
                dists[:, j] = float('inf')
            else:
                break

        for j, i in enumerate(col):
            if i == -1:
                self.update(vision_objects[j].tid, vision_objects[j], None)
            else:
                self.update(vision_objects[j].tid, vision_objects[j], replay_objects[i])

    def get_match_rids(self, rids):
        rids = np.array(rids)
        counts = Counter(rids)
        total_elements = len(rids)
        threshold_count = 30
        threshold_ratio = total_elements / 4
        
        frequent_rids = [
            rid for rid, count in counts.items()
            if count > threshold_ratio or count >= threshold_count]

        return frequent_rids

    def save_track(self, result_pd, save_alone):
        result_new = []
        for tid in self.tracks.keys():
            result_e = []
            track = self.tracks[tid]
            save_data = track.save_data
            for item in save_data:
                if item.get('replay_tid'):  # 只保留匹配到回灌结果的目标
                    result_e.append(item)
            
            threshold = 100
            if len(result_e) > 0:
                df_num = pd.DataFrame(result_e)
                # 统计 'class' 为 2，4 的次数
                count_class_2 = (df_num['class'] == 2).sum()
                count_class_4 = (df_num['class'] == 4).sum()
                if count_class_2 > 50 or count_class_4 > 50:
                    threshold = 50

            if len(result_e) >= threshold:
                result_new.extend(result_e)

        result_pd.extend(result_new)
        if save_alone:
            df_e = pd.DataFrame(result_new)
            df_e.to_csv(os.path.join(self.path, self.filename.replace('.txt', '_vision_replay_match.csv')), index=False)

    def check_touchline_flag_for_target(self, frame_index, target_id, obj_dict):
        """检查指定目标在当前帧是否压线"""
        if frame_index not in self.lane_data:
            return {
                "touchline_flag": False,
                "closest_lane_id": None,
                "closest_lane_type": None,
                "distance_to_lane": None
            }
        
        lane_list = self.lane_data[frame_index]
        if not lane_list:
            return {
                "touchline_flag": False,
                "closest_lane_id": None,
                "closest_lane_type": None,
                "distance_to_lane": None
            }
        
        results = self.touchline_flag_checker.check_all_objects_touchline_flag(obj_dict, lane_list, lane_width=0.10)
        
        if results:
            return results[0]
        else:
            return {
                "touchline_flag": False,
                "closest_lane_id": None,
                "closest_lane_type": None,
                "distance_to_lane": None
            }

    def track(self, result_pd, save_alone=False):
        # 第一次匹配：建立初始映射关系
        for frame_data in self.vision_data:
            index = frame_data["frameId"]
            self.predict()
            try:
                vision_objects = self.load_vision_objects(index)
                replay_objects = self.load_replay_objects(index)
            except:
                print(f'捕捉到读取文件1 {self.filename} 文件 {index}帧报错')
                continue
            
            try:
                dists = self.cal_dists(replay_objects, vision_objects)
            except:
                print(f'捕捉到计算代价 {self.filename} 文件 {index}帧报错')
                continue

            self.greedy_match(dists, replay_objects, vision_objects)

        # 建立映射关系
        for tid, track in self.tracks.items():
            rids = np.array([o.rid for o in track.replay_data if o])
            if len(rids):
                self.mapping[tid] = self.get_match_rids(rids)
            else:
                self.mapping[tid] = []
        
        # 清空跟踪结果，准备第二次匹配
        self.tracks = dict()
        
        # 第二次匹配：使用映射关系进行更精确的匹配
        for frame_data in self.vision_data:
            index = frame_data["frameId"]
            self.predict()
            try:
                vision_objects = self.load_vision_objects(index)
                replay_objects = self.load_replay_objects(index)
            except:
                continue
            
            try:
                dists = self.cal_dists_mapping(replay_objects, vision_objects)
            except:
                continue
            
            self.greedy_match(dists, replay_objects, vision_objects)

        # 处理压线检测
        self.process_touchline_flag_detection()
        
        self.save_track(result_pd, save_alone)

    def process_touchline_flag_detection(self):
        """处理所有匹配目标的压线检测"""
        for tid, track in self.tracks.items():
            for i, save_data in enumerate(track.save_data):
                frame_index = save_data.get('frame_index')
                target_id = save_data.get('tid')
                
                vision_frame_data = {}
                # 从vision_data中获取目标原始数据
                for frame_data in self.vision_data:
                    if frame_data["frameId"] == frame_index:
                        vision_frame_data = frame_data
                        break

                if isinstance(vision_frame_data, dict) and "Oem" in vision_frame_data:
                    vis_info = vision_frame_data["Oem"].get("visInfo", [])
                    for item in vis_info:
                        obj_attributes = item.get("obj", {}).get("objAttributes", [])
                        for obj in obj_attributes:
                            if obj.get("u8Id") == target_id:
                                # 进行压线检测
                                touchline_flag_result = self.check_touchline_flag_for_target(frame_index, target_id, obj)
                                
                                # 将压线结果添加到save_data中
                                save_data.update({
                                    "touchline_flag": touchline_flag_result["touchline_flag"],
                                    "closest_lane_id": touchline_flag_result["closest_lane_id"],
                                    "closest_lane_type": touchline_flag_result["closest_lane_type"],
                                    "distance_to_lane": touchline_flag_result["distance_to_lane"]
                                })
                                break
                        else:
                            continue
                        break

    def load_vision_objects(self, img_index):
        data = []
        vision_objects = []        
        if self.vision_keyword in ['_arcsoft_obj.txt'] or self.vision_keyword == '.txt':
            data_dict = {}
            for frame_data in self.vision_data:
                if frame_data["frameId"] == img_index:
                    data_dict = frame_data["Oem"]["visInfo"][0]["obj"]["objAttributes"]
                    break

            for data_e in data_dict:
                ObjClass = data_e.get('u8OemObjClass')
                if ObjClass in [0, 1, 2, 3, 4, 6, 7] and data_e.get('f32LatDistance'):
                    x = data_e.get('f32LongDistance', 999)
                    y = data_e.get('f32LatDistance', 999)
                    w = data_e.get('f32Width', 999)
                    h = data_e.get('f32Height', 999)
                    angle = data_e.get('f32AngleRate', 999)
                    tid = data_e.get('u8Id', 999)
                    conf = data_e.get('ExPro', 1)
                    vx = data_e.get('f32AbsoluteLongVelocity', 999)
                    vy = data_e.get('f32AbsoluteLatVelocity', 999)
                    cipv = data_e.get('s32Cipv', 999)
                    heading = data_e.get('f32Heading', 999)
                    age = data_e.get('u16ObjectAge', 999)
                    
                    vision_objects.append(VisionObject(tid, conf, w, h, x, y, vx, vy, angle, img_index, 'type'))
                    save_data = {
                        'filename': self.filename, 
                        'CIPV': cipv, 
                        'tid': tid, 
                        'age': age, 
                        'frame_index': img_index, 
                        'x': x, 
                        'y': y, 
                        'class': ObjClass, 
                        'vx': vx, 
                        'vy': vy, 
                        'heading': heading
                    }
                    vision_objects[-1].save_data = save_data

        return vision_objects

    def load_replay_objects(self, img_index):
        data = []
        replay_objects = []
        data_dict = {}
        for frame_data in self.replay_data:
            for frame_id, data in frame_data.items():
                if int(frame_id) == img_index:
                    data_dict = data
                    break

        for data_e in data_dict:
            ObjClass = data_e.get('u8OemObjClass')
            if ObjClass in [0, 1, 2, 3, 4, 6, 7] and data_e.get('f32LatDistance'):
                x = data_e.get('f32LongDistance', 999)
                y = data_e.get('f32LatDistance', 999)
                conf = data_e.get('ExPro', 1)
                vx = data_e.get('f32AbsoluteLongVelocity', 999)
                vy = data_e.get('f32AbsoluteLatVelocity', 999)
                cipv = data_e.get('s32Cipv', 999)
                heading = data_e.get('f32Heading', 999)
                age = data_e.get('u16ObjectAge', 999)
                tid = data_e.get('u8Id', 999)
                
                replay_objects.append(ReplayObject(x, y, conf, vx, vy, tid))
                save_data = {
                    'filename': self.filename,
                    'replay_CIPV': cipv,
                    'replay_tid': tid,
                    'replay_age': age,
                    'replay_frame_index': img_index,
                    'replay_x': x,
                    'replay_y': y,
                    'replay_class': ObjClass,
                    'replay_vx': vx,
                    'replay_vy': vy,
                    'replay_heading': heading
                }
                replay_objects[-1].save_data = save_data

        return replay_objects


def save_to_csv(data, output_path):
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)


def process_file(args):
    vision_file, vision_keyword, mode, map_relate, labelinfo = args
    logger.info(f'{vision_file} 实车与回灌结果匹配中...')

    filename = os.path.splitext(os.path.basename(vision_file))[0]
    label_dict = labelinfo.get(filename, {})
    frame_list = []
    for attr_data in label_dict:
        start_frame = attr_data['start_frame']
        end_frame = attr_data['end_frame']
        for frame in range(start_frame, end_frame + 1):
            if frame not in frame_list:
                frame_list.append(frame)

    try:
        model = VisionReplayHybrid(vision_file, vision_keyword, mode, map_relate, frame_list)
        result_pd = []
        model.track(result_pd, save_alone=True)
        return result_pd

    except Exception as e:
        import traceback
        logger.error(f'❌ 处理 {vision_file} 时出现异常: {e}')
        logger.error(traceback.format_exc())
        return []


def main_multiple_processes(file_all, vision_keyword, src_path, mode, map_relate, labelinfo, batch_size=1000000):
    num_processes = max(1, mp.cpu_count() - 2)
    pool = mp.Pool(processes=num_processes)
    now = datetime.now()
    formatted_time = now.strftime('%Y%m%d%H%M')
    try:
        results_iterator = pool.imap(process_file, zip(file_all, [vision_keyword] * len(file_all), [mode] * len(file_all), [map_relate] * len(file_all), [labelinfo] * len(file_all)))

        buffer = []
        batch_counter = 0
        for i, result in enumerate(tqdm(results_iterator, total=len(file_all), desc="任务正在处理..."), start=1):
            buffer.extend(result)
            if len(buffer) >= batch_size:
                batch_counter += 1
                output_filename = os.path.join(src_path, f'{formatted_time}_vision_replay_match_part{batch_counter}.csv')
                save_to_csv(buffer, output_filename)
                logger.info(f'已保存 {len(buffer)} 条数据到 {output_filename}')
                buffer.clear()

        if buffer:
            batch_counter += 1
            output_filename = os.path.join(src_path, f'{formatted_time}_vision_replay_match_part{batch_counter}.csv')
            save_to_csv(buffer, output_filename)
            logger.info(f'已保存 {len(buffer)} 条数据到 {output_filename}')
    finally:
        pool.close()
        pool.join()


def txt_path_dic(path, key_ends='.txt'):
    txt_name_path_dic = {}
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(key_ends) and '_arcsoft_obj' not in file:
                txt_path = os.path.join(root, file)
                txt_name_path_dic[file] = txt_path
    return txt_name_path_dic


if __name__ == '__main__':
    try:
        mode = False
        src_path = r'G:\chedaoxian\lixiang2'
        virtual_test_set_path = r"C:\Users\chz62985\Desktop\code\ziyi版本\真实压线视频列表.txt"
        # SpecialProject_labelfile = r"F:\Desktop\python_tools\专项\目标与车道线横向位置关系专项\result\压线检测统计结果.xlsx"
        SpecialProject_labelfile = r"C:\Users\chz62985\Desktop\code\ziyi版本\车道线与目标关系验证_真实压线数据集_表格.xlsx"

        labelinfo = LabelInfoProcessor.get_special_project_labelinfo(SpecialProject_labelfile)

        vision_keyword = '.txt'
        txt_name_path_dic = txt_path_dic(src_path, vision_keyword)
        map_relate = {}
        if os.path.isdir(virtual_test_set_path):
            for root, dirs, files in os.walk(virtual_test_set_path):
                for file in files:
                    if file.endswith('.h264') and file.replace('.h264', '.txt') in txt_name_path_dic:
                        h264_file = os.path.join(root, file)
                        map_relate[file.replace('.h264', '.txt')] = os.path.join(root, 'log', file.replace('.h264', '.txt'))
        elif os.path.isfile(virtual_test_set_path):
            with open(virtual_test_set_path, "r", encoding="utf-8", errors="ignore") as f:
                for idx, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        filename = os.path.basename(line)
                        root = os.path.dirname(line)
                        if filename.endswith('.mp4') and filename.replace('.mp4', '.txt') in txt_name_path_dic:
                            map_relate[filename.replace('.mp4', '.txt')] = os.path.join(root, 'log', filename.replace('.mp4', '.txt'))
                    except json.JSONDecodeError:
                        print(f"⚠️ 第{idx}行不是合法 JSON: {line}")

        common_keys = txt_name_path_dic.keys() & map_relate.keys()
        filter_keys = []
        for k in common_keys:
            base_k = k.replace(".txt", "") if k.endswith(".txt") else k
            if labelinfo.get(base_k):
                filter_keys.append(k)
        file_all = [txt_name_path_dic[k] for k in filter_keys]

        main_multiple_processes(file_all, vision_keyword, src_path, mode, map_relate, labelinfo)

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        tb_info = traceback.format_tb(exc_tb)
        logger.error(tb_info)
        logger.error(exc_obj)