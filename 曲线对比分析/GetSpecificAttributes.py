import json
from math import sin, cos, radians
from utils import *
from itertools import zip_longest

def is_valid(value):
    return value is not None

def extract_ego_data(file_path):
    """ 提取自车状态信息，并组织为字典格式 """
    ego_data_dict = {}

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if isinstance(data, dict):
                    for frame, ego_data in data.items():
                        frame = int(frame)  # 确保帧 ID 是整数类型
                        if "vehCan" in ego_data:
                            vis_info = ego_data.get("vehCan", [])
                            if isinstance(vis_info, list) and vis_info:
                                item = vis_info[0]  # 取第一个元素
                                ego_data_dict[frame] = {
                                    "f32Speed": item.get("f32Speed"),
                                    "f32SteerAngle": item.get("f32SteerAngle"),
                                    "f32SteerAngleSign": item.get("f32SteerAngleSign")
                                }
                        elif "carSignal" in ego_data:
                            ego_state = ego_data.get("carSignal", [])
                            if isinstance(ego_state, list) and ego_state:
                                item = ego_state[0]  # 取第一个元素
                                ego_data_dict[frame] = {
                                    "f32Speed": item.get("speed"),
                                    "f32SteerAngle": item.get("steerAngle"),
                                    "steerAngleVel": item.get("steerAngleVel")
                                }
                else:
                    print("Unsupported data format")
            except (KeyError, json.JSONDecodeError) as e:
                print(f"Error processing line: {e}")
                continue
    return ego_data_dict


def extract_objdata_arc(file_path):
    obj_data_dict = {}

    # 尝试读取 obj 文件
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='gbk', errors='ignore') as f:
            lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line or line == "{}":
            continue

        try:
            data = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"[JSON Error] {e}")
            continue

        if not isinstance(data, dict):
            print("[Warning] Unsupported line format")
            continue

        if "Oem" in data:
            frame = data["Oem"].get("u64FrameId")
            if frame > 600:
                pass
            vis_info = data["Oem"].get("visInfo", [])
            for item in vis_info:
                for attr in item.get("obj", {}).get("objAttributes", []):
                    target_id = attr.get("u8Id")
                    obj_class = attr.get("u8OemObjClass")
                    obj_type = classify_target(obj_class, "ARC")
                    # if obj_type not in ["大车", "小车", "行人", "骑行人"]:
                    #     continue

                    if frame not in obj_data_dict:
                        obj_data_dict[frame] = {}

                    obj_data_dict[frame][target_id] = {
                        "heading": attr.get("f32Heading")*57.3 if is_valid(attr.get("f32Heading")) else None,
                        "Width": attr.get("f32Width") if is_valid(attr.get("f32Width")) else None,
                        "Length": attr.get("f32Length") if is_valid(attr.get("f32Length")) else None,
                        "LongDistance": attr.get("f32LongDistance") if is_valid(attr.get("f32LongDistance")) else None,
                        "LatDistance": attr.get("f32LatDistance") if is_valid(attr.get("f32LatDistance")) else None,
                        "AbsoluteLatVelocity": attr.get("f32AbsoluteLatVelocity") if is_valid(attr.get("f32AbsoluteLatVelocity")) else None,
                        "AbsoluteLongVelocity": attr.get("f32AbsoluteLongVelocity") if is_valid(attr.get("f32AbsoluteLongVelocity")) else None,
                        "f32AbsoluteLongAcc": attr.get("f32AbsoluteLongAcc") if is_valid(attr.get("f32AbsoluteLongAcc")) else None,
                        "RelativeLatVelocity": attr.get("f32RelativeLatVelocity") if is_valid(attr.get("f32RelativeLatVelocity")) else None,
                        "RelativeLongVelocity": attr.get("f32RelativeLongVelocity") if is_valid(attr.get("f32RelativeLongVelocity")) else None,
                        "f32RelativeLongAcc": attr.get("f32RelativeLongAcc") if is_valid(attr.get("f32RelativeLongAcc")) else None,
                        "Probability": attr.get("f32ExistenceProbability") if is_valid(attr.get("f32ExistenceProbability")) else None,
                        "ObjClass": obj_class if is_valid(obj_class) else None,
                        "obj_CIPV": attr.get("s32Cipv") if is_valid(attr.get("s32Cipv")) else None,
                        "rcBox": attr.get("rcBox", {}),
                    }
        else:
            # 第二种格式：帧编号作为 key，值是目标对象列表
            ctrl_txtpath = file_path.replace('_arcsoft_obj.txt', '_arcsoft_ctrl.txt')
            try:
                with open(ctrl_txtpath, 'r', encoding='utf-8', errors='ignore') as ctrl_f:
                    ctrl_lines = ctrl_f.readlines()
            except (FileNotFoundError, UnicodeDecodeError):
                ctrl_lines = []

            for obj_line, ctrl_line in zip_longest(lines, ctrl_lines, fillvalue="{}"):
                try:
                    frame_obj_dict = json.loads(obj_line)
                    if not isinstance(frame_obj_dict, dict):
                        continue
                    
                    ctrl_data = json.loads(ctrl_line.strip())
                    PNC_Cipv_Id = None
                    if isinstance(ctrl_data, dict):
                        for frame, ctrldatalist in ctrl_data.items():
                            for ctrldata in ctrldatalist:
                                acc_info = ctrldata.get('accTargetInfo', {})
                                PNC_Cipv_Id = acc_info.get('PNC_Cipv_Id')
                                if PNC_Cipv_Id is not None:
                                    break
                    elif isinstance(ctrl_data, list):
                        for ctrldata in ctrl_data:
                            acc_info = ctrldata.get('accTargetInfo', {})
                            PNC_Cipv_Id = acc_info.get('PNC_Cipv_Id')
                            if PNC_Cipv_Id is not None:
                                break
                    
                    for frame_str, obj_list in frame_obj_dict.items():
                        try:
                            frame = int(frame_str)
                        except ValueError:
                            continue

                        for obj in obj_list:
                            target_id = obj.get("u8Id")
                            obj_class = obj.get("u8OemObjClass")
                            obj_type = classify_target(obj_class, "ARC")
                            if obj_type not in ["大车", "小车", "行人", "骑行人"]:
                                continue

                            if frame not in obj_data_dict:
                                obj_data_dict[frame] = {}

                            # 兼容存在不同字段名的概率字段
                            prob = obj.get("f32ObjectExistenceProbability") or obj.get("f32ExistenceProbability")
                            obj_data_dict[frame][target_id] = {
                                "heading": obj.get("f32Heading")*57.3 if is_valid(obj.get("f32Heading")) else None,
                                "Width": obj.get("f32Width") if is_valid(obj.get("f32Width")) else None,
                                "Length": obj.get("f32Length") if is_valid(obj.get("f32Length")) else None,
                                "LongDistance": obj.get("f32LongDistance") if is_valid(obj.get("f32LongDistance")) else None,
                                "LatDistance": obj.get("f32LatDistance") if is_valid(obj.get("f32LatDistance")) else None,
                                "AbsoluteLatVelocity": obj.get("f32AbsoluteLatVelocity") if is_valid(obj.get("f32AbsoluteLatVelocity")) else None,
                                "AbsoluteLongVelocity": obj.get("f32AbsoluteLongVelocity") if is_valid(obj.get("f32AbsoluteLongVelocity")) else None,
                                "f32AbsoluteLongAcc": obj.get("f32AbsoluteLongAcc") if is_valid(obj.get("f32AbsoluteLongAcc")) else None,
                                "RelativeLatVelocity": obj.get("f32RelativeLatVelocity") if is_valid(obj.get("f32RelativeLatVelocity")) else None,
                                "RelativeLongVelocity": obj.get("f32RelativeLongVelocity") if is_valid(obj.get("f32RelativeLongVelocity")) else None,
                                "f32RelativeLongAcc": obj.get("f32RelativeLongAcc") if is_valid(obj.get("f32RelativeLongAcc")) else None,
                                "Probability": prob if is_valid(prob) else None,
                                "ObjClass": obj_class if is_valid(obj_class) else None,
                                "obj_CIPV": 1 if target_id == PNC_Cipv_Id else 0,
                                "rcBox": obj.get("rcBox", {}),
                            }
                except Exception as e:
                    print(f"[Frame Parse Error] {e}")
            break  # 只处理一次整批 lines

    return obj_data_dict

def extract_objdata_fusion(file_path):
    obj_data_dict = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='gbk', errors='ignore') as f:
            lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"[JSON Error] {e}")
            continue
        if not isinstance(data, dict):
            print("[Warning] Unsupported line format")
            continue

        frame = data["Oem"].get("u64FrameId")
        vis_info = data["Oem"].get("visInfo", [])
        for item in vis_info:
            for attr in item.get("fusionObj", {}).get("objAttributes", []):
                target_id = attr.get("u8Id")
                if frame not in obj_data_dict:
                    obj_data_dict[frame] = {}

                obj_data_dict[frame][target_id] = {
                    "heading": attr.get("f32Heading")*57.3,
                    "Width": attr.get("f32Width"),
                    "Length": attr.get("f32Length"),
                    "LongDistance": attr.get("f32LongDistance"),
                    "LatDistance": attr.get("f32LatDistance"),
                    "AbsoluteLatVelocity": attr.get("f32AbsoluteLatVelocity"),
                    "AbsoluteLongVelocity": attr.get("f32AbsoluteLongVelocity"),
                    "RelativeLatVelocity": attr.get("f32RelativeLatVelocity"),
                    "RelativeLongVelocity": attr.get("f32RelativeLongVelocity"),
                    "RelativeLongAcc": attr.get("f32RelativeLongAcc"),
                    "u64VisionAge": attr.get("u64VisionAge"),
                    "u64RadarAge": attr.get("u64RadarAge"),
                    "u64VisisonId": attr.get("u64VisisonId"),
                    "u64RadarId": attr.get("u64RadarId"),
                    "u8CipvFlag": attr.get("u8CipvFlag"),
                    "u8Id": attr.get("u8Id"),
                }
        
    return obj_data_dict


def find_most_frequent_cipv_target(obj_data_dict, start_frame, end_frame):
    """
    在指定的帧范围内，找到 obj_CIPV 为 1 最多的目标 ID。
    
    :param obj_data_dict: 数据字典，包含目标信息。
    :param start_frame: 开始帧
    :param end_frame: 结束帧
    :return: obj_CIPV 为 1 的目标 ID 最多的目标 ID
    """
    target_cipv_count = {}

    # 遍历 obj_data_dict 中的所有帧
    for frame, objects in obj_data_dict.items():
        if frame < start_frame or frame > end_frame:
            continue  # 跳过不在帧范围内的数据

        for target_id, attributes in objects.items():
            obj_CIPV = attributes.get("obj_CIPV", None)

            # 如果 obj_CIPV 为 1，统计该目标 ID
            if obj_CIPV == 1:
                if target_id not in target_cipv_count:
                    target_cipv_count[target_id] = 0
                target_cipv_count[target_id] += 1

    if target_cipv_count:
        # 找到 obj_CIPV 为 1 出现次数最多的目标 ID
        most_frequent_target_id = max(target_cipv_count, key=target_cipv_count.get)
        return most_frequent_target_id
    else:
        return None  # 如果没有目标的 obj_CIPV 为 1，返回 None


def extract_objdata_J3(file_path):
    """
    从指定文件中提取数据，根据帧范围和目标 ID 进行筛选
    """
    obj_data_dict = {}

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if isinstance(data, dict):
                    for frame, objects in data.items():
                        frame = int(frame)  # 确保帧 ID 是整数类型
                        for obj in objects:
                            target_id = obj.get("id")
                            heading = obj.get("obstacle_angle")
                            Width = obj.get("obstacle_width")
                            Length = obj.get("obstacle_length")
                            longDistance = obj.get("obstacle_pos_x")
                            latDistance = obj.get("obstacle_pos_y")
                            RelativeLatVelocity = obj.get("obstacle_vel_y")
                            RelativeLongVelocity = obj.get("obstacle_vel_x")
                            absoluteLatVelocity = obj.get("obstacle_vel_abs_y")
                            absoluteLongVelocity = obj.get("obstacle_vel_abs_x")
                            ObjClass = obj.get("type")
                            obj_type = classify_target(ObjClass, "J3")
                            if obj_type not in ["车辆", "行人", "骑行人"]:
                                continue

                            if frame not in obj_data_dict:
                                obj_data_dict[frame] = {}
                            
                            obj_data_dict[frame][target_id] = {
                                "heading": heading*57.3 if is_valid(heading) else None,
                                "Width": Width if is_valid(Width) else None,
                                "Length": Length if is_valid(Length) else None,
                                "LongDistance": longDistance if is_valid(longDistance) else None,
                                "LatDistance": latDistance if is_valid(latDistance) else None,
                                "RelativeLatVelocity": RelativeLatVelocity if is_valid(RelativeLatVelocity) else None,
                                "RelativeLongVelocity": RelativeLongVelocity if is_valid(RelativeLongVelocity) else None,
                                "AbsoluteLatVelocity": absoluteLatVelocity if is_valid(absoluteLatVelocity) else None,
                                "AbsoluteLongVelocity": absoluteLongVelocity if is_valid(absoluteLongVelocity) else None,
                                "ObjClass": ObjClass if is_valid(ObjClass) else None
                            }
                else:
                    print("Unsupported data format")
            except (KeyError, json.JSONDecodeError) as e:
                print(f"Error processing line: {e}")
                continue

    return obj_data_dict


def extract_data_eq4(file_path, start_frame, end_frame):
    """
    从指定文件中提取数据，根据帧范围和目标 ID 进行筛选
    """
    obj_data_dict = {}

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                
                if isinstance(data, dict):
                    for frame, objects in data.items():
                        frame = int(frame)  # 确保帧 ID 是整数类型
                        if start_frame <= frame <= end_frame:
                            for obj in objects:
                                target_id = obj.get("id")
                                heading = obj.get("heading")*57.3
                                Width = obj.get("width")
                                Length = obj.get("length")
                                longDistance = obj.get("longDistance")
                                latDistance = obj.get("latDistance")
                                AbsoluteLatVelocity = obj.get("absoluteLatVelocity")
                                AbsoluteLongVelocity = obj.get("absoluteLongVelocity")
                                ObjClass = obj.get("classification")
                                obj_type = classify_target(ObjClass, "J3")
                                if obj_type not in ["大车", "小车", "行人", "骑行人"]:
                                    continue

                                if frame not in obj_data_dict:
                                    obj_data_dict[frame] = {}
                                
                                obj_data_dict[frame][target_id] = {
                                    "heading": heading*57.3 if is_valid(heading) else None,
                                    "Width": Width if is_valid(Width) else None,
                                    "Length": Length if is_valid(Length) else None,
                                    "LongDistance": longDistance if is_valid(longDistance) else None,
                                    "LatDistance": latDistance if is_valid(latDistance) else None,
                                    "AbsoluteLatVelocity": AbsoluteLatVelocity if is_valid(AbsoluteLatVelocity) else None,
                                    "AbsoluteLongVelocity": AbsoluteLongVelocity if is_valid(AbsoluteLongVelocity) else None,
                                    "ObjClass": ObjClass if is_valid(ObjClass) else None
                                }
                else:
                    print("Unsupported data format")
            except (KeyError, json.JSONDecodeError) as e:
                print(f"Error processing line: {e}")
                continue

    return obj_data_dict

def extract_data_RT3003(file_path):
    """
    从指定文件中提取数据，根据帧范围和目标 ID 进行筛选
    """
    if not os.path.exists(file_path):
        return {}
    obj_data_dict = {}

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                
                if isinstance(data, dict):
                    for frame, objects in data.items():
                        frame = int(frame)  # 确保帧 ID 是整数类型
                        
                        # 初始化目标数据存储
                        if frame not in obj_data_dict:
                            obj_data_dict[frame] = []
                        
                        for obj in objects:
                            try:
                                if obj.get("Range1PosForward") is not None:
                                    longDistance = obj.get("Range1PosForward") + 4
                                    latDistance = -obj.get("Range1PosLateral")
                                    Rel_longVel = obj.get("Range1VelForward")
                                    Rel_latVel = -obj.get("Range1VelLateral")
                                else:
                                    longDistance = None
                                    latDistance = None
                                    Rel_longVel = None
                                    Rel_latVel = None
                                
                                # if obj.get("Range1PosForward") is not None:
                                #     longDistance = obj.get("Range1LocalDeltaX") + 4
                                #     latDistance = obj.get("Range1LocalDeltaY")
                                #     Rel_longVel = obj.get("Range1VelForward")
                                #     Rel_latVel = -obj.get("Range1VelLateral")
                                # else:
                                #     longDistance = None
                                #     latDistance = None
                                #     Rel_longVel = None
                                #     Rel_latVel = None


                                # 只存储有效的数据
                                if is_valid(longDistance) and is_valid(latDistance):
                                    obj_data_dict[frame].append({
                                        "LongDistance": longDistance,
                                        "LatDistance": latDistance,
                                        "RelativeLongVelocity": Rel_longVel,
                                        "RelativeLatVelocity": Rel_latVel,
                                    })
                            except KeyError as e:
                                print(f"KeyError processing object: {e}")
                                continue
                else:
                    print("Unsupported data format")
            except (KeyError, json.JSONDecodeError) as e:
                print(f"Error processing line: {e}")
                continue

    return obj_data_dict



def extract_data_ars(file_path):
    """
    从指定文件中提取数据，根据帧范围和目标 ID 进行筛选
    """
    obj_data_dict = {}

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                
                if isinstance(data, dict):
                    for frame, objects in data.items():
                        frame = int(frame)  # 确保帧 ID 是整数类型
                        for obj in objects:
                            try:
                                target_id = obj.get("id")
                                heading = obj.get("heading")*57.3
                                Width = obj.get("width")
                                Length = obj.get("length")
                                longDistance = obj.get("pos_lon")
                                latDistance = -obj.get("pos_lat")
                                RelativeLatVelocity = obj.get("vel_lat")
                                RelativeLongVelocity = obj.get("vel_lon")
                                ObjClass = obj.get("class")

                                if frame not in obj_data_dict:
                                    obj_data_dict[frame] = {}
                                
                                obj_data_dict[frame][target_id] = {
                                    "heading": heading*57.3 if is_valid(heading) else None,
                                    "Width": Width if is_valid(Width) else None,
                                    "Length": Length if is_valid(Length) else None,
                                    "LongDistance": longDistance if is_valid(longDistance) else None,
                                    "LatDistance": latDistance if is_valid(latDistance) else None,
                                    "RelativeLatVelocity": RelativeLatVelocity if is_valid(RelativeLatVelocity) else None,
                                    "RelativeLongVelocity": RelativeLongVelocity if is_valid(RelativeLongVelocity) else None,
                                    "ObjClass": ObjClass if is_valid(ObjClass) else None
                                }

                            except:
                                continue
                else:
                    print("Unsupported data format")
            except (KeyError, json.JSONDecodeError) as e:
                print(f"Error processing line: {e}")
                continue

    return obj_data_dict


def extract_lane_data_arc(file_path):
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
                            f32Probability =  attr.get("f32Probability", None)
                            f32StartX = attr.get("f32StartX", None)
                            f32StartY = attr.get("f32StartY", None)
                            f32EndX = attr.get("f32EndX", None)
                            f32EndY = attr.get("f32EndY", None)
                            s32LaneTypeClass = attr.get("s32LaneTypeClass", None)


                            if frame not in lane_data_dict:
                                lane_data_dict[frame] = {}

                            lane_data_dict[frame][lane_id] = {
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
                            }
                else:
                    for frame, lane_attributes in data.items():
                        frame = int(frame)  # 确保帧 ID 是整数类型
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
                            f32Probability =  attr.get("f32Probability", None)
                            f32StartX = attr.get("f32StartX", None)
                            f32StartY = attr.get("f32StartY", None)
                            f32EndX = attr.get("f32EndX", None)
                            f32EndY = attr.get("f32EndY", None)
                            s32LaneTypeClass =  attr.get("s32LaneTypeClass", None)

                            if frame not in lane_data_dict:
                                lane_data_dict[frame] = {}

                            lane_data_dict[frame][lane_id] = {
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
                                "s32LaneTypeClass": s32LaneTypeClass if is_valid(s32LaneTypeClass) else None
                            }

            else:
                print("Unsupported data format")
        except (KeyError, json.JSONDecodeError) as e:
            print(f"Error processing line: {e}")
            continue

    return lane_data_dict


# 提取数据函数
def extract_lane_data_J3(file_path):
    lane_data_dict = {}
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                if isinstance(data, dict):
                   for frame, lane_attributes in data.items():
                        frame = int(frame)  # 确保帧 ID 是整数类型
                        for attr in lane_attributes:
                            lane_id = attr.get("id", None)
                            C0_values = attr.get("coeffs_0", None)
                            C1_values = attr.get("coeffs_1", None)
                            C2_values = attr.get("coeffs_2", None)
                            C3_values = attr.get("coeffs_3", None)
                            u8OemSide = attr.get("u8OemSide", None)
                            u8OemType = attr.get("type_type", None)
                            u8OemColor = attr.get("type_color", None)
                            f32Probability =  attr.get("conf", None)
                            s32LaneTypeClass =  attr.get("s32LaneTypeClass", None)

                            
                            # f32StartX = attr.get("end_points")["x"]
                            # f32StartY = attr.get("end_points")["y"]
                            # f32EndX = attr.get("end_points")["x"]
                            # f32EndY = attr.get("end_points")["y"]

                            if frame not in lane_data_dict:
                                lane_data_dict[frame] = {}

                            lane_data_dict[frame][lane_id] = {
                                "f32LineC0": C0_values if is_valid(C0_values) else None,
                                "f32LineC1": C1_values if is_valid(C1_values) else None,
                                "f32LineC2": C2_values if is_valid(C2_values) else None,
                                "f32LineC3": C3_values if is_valid(C3_values) else None,
                                "f32LineC3": C3_values if is_valid(C3_values) else None,
                                "u8OemSide": u8OemSide if is_valid(u8OemSide) else None,
                                "u8OemType": u8OemType if is_valid(u8OemType) else None,
                                "u8OemColor": u8OemColor if is_valid(u8OemColor) else None,
                                "f32Probability": f32Probability if is_valid(f32Probability) else None,
                                # "f32StartX": f32StartX if is_valid(f32StartX) else None,
                                # "f32StartY": f32StartY if is_valid(f32StartY) else None,
                                # "f32EndX": f32EndX if is_valid(f32EndX) else None,
                                # "f32EndY": f32EndY if is_valid(f32EndY) else None,
                                "s32LaneTypeClass": s32LaneTypeClass if is_valid(s32LaneTypeClass) else None
                            }


                else:
                    print("Unsupported data format")
            except (KeyError, json.JSONDecodeError) as e:
                print(f"Error processing line: {e}")
                continue

    return lane_data_dict