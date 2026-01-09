import json
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.patches import Rectangle, Polygon


def find_h264_files(folder_path):
    """
    查找指定文件夹下的所有.h264文件。
    :param folder_path: 文件夹路径
    :return: .h264文件路径列表
    """
    h264_files = {}
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".h264"):
                filename = os.path.splitext(file)[0]
                filedirname = os.path.splitext(os.path.basename(os.path.dirname(root)))[0]
                if filename not in h264_files:
                    h264_files[filename] = root
                    
    return h264_files

def find_txt_files(folder_path):
    """
    查找指定文件夹下的所有.h264文件。
    :param folder_path: 文件夹路径
    :return: .h264文件路径列表
    """
    txt_files = {}
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".txt"):
                filename = os.path.splitext(file)[0]
                filedirname = os.path.splitext(os.path.basename(os.path.dirname(root)))[0]
                if filename not in txt_files:
                    txt_files[filename] = os.path.join(root, file)
    return txt_files

def save_result(result_fliepath, attr):
    with open(result_fliepath, 'a', encoding='utf-8') as f:
        f.write(json.dumps(attr, ensure_ascii=False) + '\n')
 
def list_unprocessed_aeb_files(folder_path):
    """查找所有的待处理的素材所在路径"""
    h264_files_list = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.h264'):
                h264_files_list.append(os.path.join(root, file))
    return h264_files_list

def classify_target(u8OemObjClass, compete_type):
    if compete_type == 'ARC':
        """将 u8OemObjClass 转换为类别名称"""
        if u8OemObjClass in {0, 3, 6, 7}:
            return "大车"
        elif u8OemObjClass == 1:
            return "小车"
        elif u8OemObjClass == 2:
            return "行人"
        elif u8OemObjClass == 4:
            return "骑行人"
        else:
            return "交通锥"
        
    elif compete_type == "EQ4":
        if u8OemObjClass == 4:
            return "行人"
        elif u8OemObjClass == 3 or u8OemObjClass == 5: #5为骑自行车，3为骑摩托车
            return "骑行人"
        elif u8OemObjClass == 1:
            return "小车"
        elif u8OemObjClass == 2:
            return "大车"
        
    elif compete_type == "J3":
        """将 u8OemObjClass 转换为类别名称"""
        if u8OemObjClass == 2:
            return "行人"
        elif u8OemObjClass == 3:
            return "骑行人"
        elif u8OemObjClass == 1:
            return "车辆"
        else:
            return "交通锥"

# 计算目标与车道线的相对距离
def Calculate_obj_lane_distance(long_distance, lat_distance, lane):
    if lane is None:
        return None
    f32EndX = lane.get("f32EndX")
    c0 = lane.get("f32LineC0")
    c1 = lane.get("f32LineC1")
    c2 = lane.get("f32LineC2")
    c3 = lane.get("f32LineC3")
    lat_dis = c0 + c1 * long_distance + c2 * long_distance**2 + c3 * long_distance**3
    return lat_dis - lat_distance

# 获取车道线位置
def GetLaneLocation(lane, compete_type):
    if compete_type == "ARC":
        lane_side = {
            (1, 1): "Lane_L1",
            (1, 2): "Lane_R1",
            (1, 3): "Lane_L2",
            (1, 4): "Lane_R2",
            (1, 5): "Lane_L3",
            (1, 6): "Lane_R3",
            (1, 7): "Lane_L4",
            (1, 8): "Lane_R4",
            (2, 1): "Lane_L1",
            (2, 2): "Lane_R1",
            (2, 3): "Lane_L2",
            (2, 4): "Lane_R2",
            (2, 5): "Lane_L3",
            (2, 6): "Lane_R3",
            (2, 7): "Lane_L4",
            (2, 8): "Lane_R4",
            (3, 1): "Edge_L",
            (3, 2): "Edge_R"
        }
        s32LaneTypeClass = lane.get('s32LaneTypeClass')
        u8OemSide = lane.get('u8OemSide')
        lane_type = lane_side.get((s32LaneTypeClass, u8OemSide), 'un_known')

    elif compete_type == "J3":
        lane_side = {
            (1, 1): "Lane_L1",
            (1, 2): "Lane_R1",
            (1, 3): "Lane_L2",
            (1, 4): "Lane_R2",
            (1, 5): "Lane_L3",
            (1, 6): "Lane_R3",
            (1, 7): "Lane_L4",
            (1, 8): "Lane_R4",
            (2, 1): "Lane_L1",
            (2, 2): "Lane_R1",
            (2, 3): "Lane_L2",
            (2, 4): "Lane_R2",
            (2, 5): "Lane_L3",
            (2, 6): "Lane_R3",
            (2, 7): "Lane_L4",
            (2, 8): "Lane_R4",
            (3, 1): "Edge_L",
            (3, 2): "Edge_R"
        }
        s32LaneTypeClass = lane.get('s32LaneTypeClass')
        u8OemSide = lane.get('u8OemSide')
        lane_type = lane_side.get((s32LaneTypeClass, u8OemSide), 'un_known')

    return lane_type
    

def get_continuous_target_frame_ranges(obj_data_dict):
    """
    获取每个 target_id 在 obj_data_dict 中连续出现的起始帧和结束帧（可能有多个连续段），
    并记录每段起始帧时的 LongDistance 和 LatDistance。

    参数:
        obj_data_dict (dict): 数据字典，结构为 obj_data_dict[frame][target_id] = {...}

    返回:
        dict: 每个 target_id 对应多个连续的帧范围和初始帧位置信息
              {
                  target_id1: [
                      {
                          "start_frame": x,
                          "end_frame": y,
                          "initial_LongDistance": xxx,
                          "initial_LatDistance": yyy
                      },
                      {...}
                  ],
                  target_id2: [...],
                  ...
              }
    """
    target_frame_ranges = {}  # 最终结果
    active_targets = {}       # 当前追踪的 target_id -> {'start_frame': x, 'LongDistance': xx, 'LatDistance': yy}

    sorted_frames = sorted(obj_data_dict.keys())

    for frame_idx, frame in enumerate(sorted_frames):
        frame_targets = obj_data_dict[frame]
        current_target_ids = set(frame_targets.keys())

        # 处理已经不连续的目标（结束段）
        for target_id in list(active_targets.keys()):
            if target_id not in current_target_ids:
                # 当前 target_id 已不在当前帧，结束段
                active_target_data = active_targets.pop(target_id)
                start_frame = active_target_data['start_frame']

                if target_id not in target_frame_ranges:
                    target_frame_ranges[target_id] = []

                target_frame_ranges[target_id].append({
                    "start_frame": start_frame,
                    "end_frame": sorted_frames[frame_idx - 1],
                    "ObjClass": active_target_data.get('ObjClass'),
                    "initial_LongDistance": active_target_data.get('LongDistance'),
                    "initial_LatDistance": active_target_data.get('LatDistance')
                })

        # 处理新的活跃目标（新段开始）
        for target_id in current_target_ids:
            if target_id not in active_targets:
                target_data = frame_targets[target_id]

                # 获取初始帧的 LongDistance 和 LatDistance
                long_distance = target_data.get("LongDistance")
                lat_distance = target_data.get("LatDistance")
                ObjClass = target_data.get("ObjClass")

                active_targets[target_id] = {
                    "start_frame": frame,
                    "LongDistance": long_distance,
                    "LatDistance": lat_distance,
                    "ObjClass": ObjClass
                }

    # 收尾：处理还活跃在最后一帧的目标
    for target_id, active_target_data in active_targets.items():
        start_frame = active_target_data['start_frame']

        if target_id not in target_frame_ranges:
            target_frame_ranges[target_id] = []

        target_frame_ranges[target_id].append({
            "start_frame": start_frame,
            "end_frame": sorted_frames[-1],
            "ObjClass": active_target_data.get('ObjClass'),
            "initial_LongDistance": active_target_data.get('LongDistance'),
            "initial_LatDistance": active_target_data.get('LatDistance')
        })

    return target_frame_ranges


def check_collision_with_visualization(Objdata, show_bev=False):
    """
    检查目标车辆是否与自车范围重叠，并可选择显示BEV结果
    
    参数:
        ArcObjdata: 包含目标车辆信息的字典
        ego_range_long: 自车纵向范围(默认5.03米)
        ego_range_lat: 自车横向范围(默认1.96米)
        show_bev: 是否显示BEV可视化结果(默认False)
    
    返回:
        bool: True表示有碰撞, False表示无碰撞
        fig: matplotlib图形对象(如果show_bev=True)
    """
    # 提取目标车辆数据
    heading = Objdata.get("heading")  # 朝向角(弧度)
    if heading is None:
        heading = 0.0
    heading = -heading

    length = Objdata.get("Length")    # 车辆长度
    width = Objdata.get("Width")      # 车辆宽度
    lat_dist = Objdata.get("LatDistance")  # 横向距离(左侧为正)
    long_dist = Objdata.get("LongDistance")  # 纵向距离(前方为正)
    if heading is None or length is None or width is None or lat_dist is None or long_dist is None:
        # print("当前帧出现数据异常, 无法判断")
        return None
    
    # 计算目标车辆的四个角点(相对于车辆中心)
    half_len = length
    half_wid = width / 2
    
    # 自车范围(以自车中心为原点)
    # 坐标系: 正前方为Y正方向，左侧为X正方向
    ego_corners = np.array([
        [-0.98, -1.135], # 左后
        [-0.98, 3.895],  # 左前
        [0.98, 3.895],   # 右前
        [0.98, -1.135]   # 右后
    ])
    
    # 目标车辆未旋转时的角点(相对于车辆中心)
    obj_corners_local = np.array([
        [-half_wid, 0],  # 左后
        [-half_wid, half_len],   # 左前
        [half_wid, half_len],    # 右前
        [half_wid, 0]     # 右后
    ])
    
    # 旋转矩阵
    rot_mat = np.array([
        [math.cos(heading), -math.sin(heading)],
        [math.sin(heading), math.cos(heading)]
    ])
    
    # 先旋转后平移目标车辆角点
    obj_corners_rotated = np.dot(obj_corners_local, rot_mat.T)
    obj_corners_global = obj_corners_rotated + np.array([lat_dist, long_dist])
    
    # 使用分离轴定理(SAT)检测碰撞
    def project(poly, axis):
        dots = np.dot(poly, axis)
        return min(dots), max(dots)
    
    def overlap(proj1, proj2):
        return not (proj1[1] < proj2[0] or proj2[1] < proj1[0])
    
    # 多边形的边
    polygons = [ego_corners, obj_corners_global]
    edges = []
    
    for poly in polygons:
        for i in range(len(poly)):
            edge = poly[(i+1)%len(poly)] - poly[i]
            normal = np.array([-edge[1], edge[0]])
            normal = normal / np.linalg.norm(normal)  # 单位化
            edges.append(normal)
    
    # 检查所有分离轴
    collision = True
    for axis in edges:
        proj1 = project(ego_corners, axis)
        proj2 = project(obj_corners_global, axis)
        
        if not overlap(proj1, proj2):
            collision = False
            break
    
    # 如果需要显示BEV结果
    if show_bev and collision:
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # 绘制自车范围
        ego_poly = Polygon(ego_corners, closed=True,
                          edgecolor='r', facecolor='red', alpha=0.3)
        ax.add_patch(ego_poly)
        
        # 绘制目标车辆
        obj_poly = Polygon(obj_corners_global, closed=True,
                          edgecolor='b', facecolor='blue', alpha=0.3)
        ax.add_patch(obj_poly)
        
        # 设置图形属性
        min_x = min(np.min(ego_corners[:, 0]), np.min(obj_corners_global[:, 0])) - 1
        max_x = max(np.max(ego_corners[:, 0]), np.max(obj_corners_global[:, 0])) + 1
        min_y = min(np.min(ego_corners[:, 1]), np.min(obj_corners_global[:, 1])) - 1
        max_y = max(np.max(ego_corners[:, 1]), np.max(obj_corners_global[:, 1])) + 1
        
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
        ax.invert_xaxis()  # 翻转X轴使左侧显示在左侧
        ax.set_aspect('equal')
        ax.grid(True)
        ax.set_xlabel('Lateral Distance (m) - Left Positive, Right Negative')
        ax.set_ylabel('Longitudinal Distance (m) - Forward Positive')
        ax.set_title(f'BEV Vehicle Collision Check (Collision: {collision})')
        
        # 添加坐标轴指示
        ax.arrow(0, 0, 0, 1, head_width=0.1, head_length=0.2, fc='g', ec='g', label='Forward')
        ax.arrow(0, 0, -1, 0, head_width=0.1, head_length=0.2, fc='m', ec='m', label='Left')
        ax.legend()
        plt.show()
            
    return collision

