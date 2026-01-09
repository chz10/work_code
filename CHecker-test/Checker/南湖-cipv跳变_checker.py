import os
import json
import argparse
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


def align_trajectories(traj1, traj2):
    traj2_frames = {list(point.keys())[0] for point in traj2}
    aligned_traj1 = []
    for point in traj1:
        frame_id = list(point.keys())[0]
        if frame_id in traj2_frames:
            aligned_traj1.append(point)
    aligned_traj1.sort(key=lambda p: list(p.keys())[0])
    aligned_traj2 = sorted(traj2, key=lambda p: list(p.keys())[0])
    return aligned_traj1, aligned_traj2


def trajectory_similarity(traj1, traj2):
    if not traj1 or not traj2:
        raise ValueError("输入的轨迹为空")

    aligned_traj1, aligned_traj2 = align_trajectories(traj1, traj2)
    if len(aligned_traj1) != len(aligned_traj2):
        raise ValueError("对齐后轨迹长度不一致，请检查数据")

    def get_values(traj):
        return [list(point.values())[0] for point in traj]

    values1 = get_values(aligned_traj1)
    values2 = get_values(aligned_traj2)
    pos1 = [[v[0], v[1]] for v in values1]
    pos2 = [[v[0], v[1]] for v in values2]
    vel1 = [[v[2], v[3]] for v in values1]
    vel2 = [[v[2], v[3]] for v in values2]
    size1 = [[v[4], v[5], v[6]] for v in values1]
    size2 = [[v[4], v[5], v[6]] for v in values2]

    pos_dist, _ = fastdtw(pos1, pos2, dist=euclidean)
    vel_dist, _ = fastdtw(vel1, vel2, dist=euclidean)
    size_dist, _ = fastdtw(size1, size2, dist=euclidean)
    n = len(values2)
    normalized_pos = pos_dist / n
    normalized_vel = vel_dist / n
    normalized_size = size_dist / n
    weights = {"pos": 1.0, "vel": 0, "size": 0}
    total_score = (
        weights["pos"] * normalized_pos +
        weights["vel"] * normalized_vel +
        weights["size"] * normalized_size
    )
    return total_score


def find_best_match(GT_track, Algo_tracks):
    similarity_scores = []
    gt_traj = list(GT_track.values())[0]
    for algo_id, algo_traj in Algo_tracks.items():
        try:
            dist = trajectory_similarity(gt_traj, algo_traj)
        except Exception:
            dist = float('inf')
        start_frame_id = list(algo_traj[0].keys())[0]
        end_frame_id = list(algo_traj[-1].keys())[0]
        similarity_scores.append((algo_id, dist, start_frame_id, end_frame_id))
    similarity_scores.sort(key=lambda x: x[1])
    return similarity_scores


def find_covering_detections(detections, gt_frame_id_range, min_gap=20):
    gt_start, gt_end = gt_frame_id_range
    blanks = [(gt_start, gt_end)]
    covering_detections = []
    for det in detections:
        if det[1] >= 15:
            break
        _, _, start, end = det
        new_blanks = []
        for blank in blanks:
            blank_start, blank_end = blank
            if abs(blank_start - start) < 10 and abs(blank_end - end) < 10:
                covering_detections.append(det)
            elif abs(blank_start - start) < 10 and blank_end - end > 10:
                new_blanks.append((end, blank_end))
                covering_detections.append(det)
            elif abs(blank_end - end) < 10 and start - blank_start > 10:
                new_blanks.append((blank_start, start))
                covering_detections.append(det)
            elif blank_end - end > 10 and start - blank_start > 10:
                new_blanks.append((blank_start, start))
                new_blanks.append((end, blank_end))
                covering_detections.append(det)
            else:
                new_blanks.append(blank)
                continue

        filtered = [(s, e) for s, e in new_blanks if e - s >= min_gap]
        merged = []
        for interval in sorted(filtered, key=lambda x: x[0]):
            if not merged:
                merged.append(interval)
            else:
                last = merged[-1]
                if interval[0] <= last[1]:
                    merged[-1] = (last[0], max(last[1], interval[1]))
                else:
                    merged.append(interval)
        blanks = merged
        if not blanks:
            break
    return covering_detections


def get_recharge_value(best_match_file_data, checker_type):
    signal_type_map = {
        '横向距离跳变': 'f32LatDistance',
        '纵向距离跳变': 'f32LongDistance',
        '横向速度跳变': 'f32AbsoluteLatVelocity',
        '纵向速度跳变': 'f32AbsoluteLongVelocity',
        '横向加速度跳变': 'f32AbsoluteLatAcc',
        '纵向加速度跳变': 'f32AbsoluteLongAcc',
        '朝向角跳变': 'f32Heading',
        '类型跳变': 'u8OemObjClass',
        'cipv跳变': 's32Cipv',
    }
    signal_type = signal_type_map.get(checker_type)
    if isinstance(signal_type, str):
        return best_match_file_data.get(signal_type)
    return None


def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        file_datas = [json.loads(line.strip()) for line in f if line.strip()]
    return file_datas


def checker(pseudo_true_value_path, recharge_res_path, checker_save_path, checker_type):
    pseudo_true_values = read_file(pseudo_true_value_path)
    if not pseudo_true_values:
        raise ValueError("伪真值文件为空")

    start_end_frame = (pseudo_true_values[0]["frame_id"], pseudo_true_values[-1]["frame_id"])
    recharge_res = read_file(recharge_res_path)

    pseudo_map_attribute = {
        '横向距离跳变': 'latDistance',
        '纵向距离跳变': 'longDistance',
        '横向速度跳变': 'absolute_lateral_velocity',
        '纵向速度跳变': 'absolute_longitudinal_velocity',
        '横向加速度跳变': 'absolute_lateral_acceleration',
        '纵向加速度跳变': 'absolute_longitudinal_acceleration',
        '朝向角跳变': 'heading',
        '类型跳变': 'obstacles_type',
        'cipv跳变': 'cipv',
    }
    pseudo_attribute = pseudo_map_attribute[checker_type]

    dt_objs_dict = {}
    gt_objs_dict = {}
    pseudo_frame_type = {}

    for pseudo_true_value in pseudo_true_values:
        frame = pseudo_true_value["frame_id"]
        obj = pseudo_true_value["generated_objects"][0] if pseudo_true_value.get("generated_objects") else {}
        pseudo_true_value.update(obj)
        gt_objs_dict[frame] = pseudo_true_value
        pseudo_frame_type[frame] = pseudo_true_value.get(pseudo_attribute)

        frame_idx = int(frame)
        if frame_idx >= len(recharge_res):
            continue
        frame_data = recharge_res[frame_idx]
        if not frame_data:
            continue

        # 安全访问嵌套字段
        try:
            vis_info = frame_data["Oem"]["visInfo"]
            if not vis_info or not isinstance(vis_info, list):
                continue
            obj_attrs = vis_info[0]["obj"]["objAttributes"]
            if not isinstance(obj_attrs, list):
                continue
        except (KeyError, IndexError, TypeError):
            continue

        recharge_objs = [obj for obj in obj_attrs if obj.get("u8OemObjClass") not in [5]]
        for recharge_obj in recharge_objs:
            dt_id = recharge_obj.get('u8Id')
            if dt_id is None:
                continue
            if dt_id not in dt_objs_dict:
                dt_objs_dict[dt_id] = []
            dt_objs_dict[dt_id].append({frame: recharge_obj})

    # 构建轨迹字典（用于匹配）
    dt_id_value_dict = {}
    gt_id_value_dict = {}
    for frame, obj in gt_objs_dict.items():
        gt_id = obj.get('ID')
        if gt_id is not None:
            gt_val = [
                obj.get('longDistance', 0),
                obj.get('latDistance', 0),
                obj.get('absolute_longitudinal_velocity', 0),
                obj.get('absolute_lateral_velocity', 0),
                obj.get('length', 0),
                obj.get('width', 0),
                obj.get('height', 0),
            ]
            if gt_id not in gt_id_value_dict:
                gt_id_value_dict[gt_id] = []
            gt_id_value_dict[gt_id].append({frame: gt_val})

    for dt_id, obj_list in dt_objs_dict.items():
        dt_id_value_dict[dt_id] = []
        for item in obj_list:
            frame = list(item.keys())[0]
            obj = item[frame]
            dt_val = [
                obj.get('f32LongDistance', 0),
                obj.get('f32LatDistance', 0),
                obj.get('f32AbsoluteLongVelocity', 0),
                obj.get('f32AbsoluteLatVelocity', 0),
                obj.get('f32Length', 0),
                obj.get('f32Width', 0),
                obj.get('f32Height', 0),
            ]
            dt_id_value_dict[dt_id].append({frame: dt_val})

    best_match = find_best_match(gt_id_value_dict, dt_id_value_dict) if gt_id_value_dict and dt_id_value_dict else []
    matched_groups = find_covering_detections(best_match, start_end_frame, min_gap=20)
    matched_groups = sorted(matched_groups, key=lambda x: (x[2], x[3]))

    recharge_frame_type = {}
    dominant_id_list = []
    for matched_group in matched_groups:
        dt_id = matched_group[0]
        dominant_id_list.append(dt_id)
        for item in dt_objs_dict.get(dt_id, []):
            frame = list(item.keys())[0]
            cipv_val = item[frame].get('s32Cipv')
            recharge_frame_type[frame] = cipv_val

    total_frames = len(pseudo_frame_type)
    if total_frames == 0:
        pseudo_no_type_recharge_rate = 0.0
        pseudo_no_type_recharge = 0
        pseudo_no_type_recharge_list = []
    else:
        pseudo_no_type_recharge = 0
        pseudo_no_type_recharge_list = []
        for frame_num in pseudo_frame_type:
            gt_val = pseudo_frame_type[frame_num]
            dt_val = recharge_frame_type.get(frame_num)
            if gt_val != dt_val:
                pseudo_no_type_recharge += 1
                pseudo_no_type_recharge_list.append(frame_num)
        pseudo_no_type_recharge_rate = round((total_frames - pseudo_no_type_recharge) / total_frames, 2)

    save_file_path = os.path.join(checker_save_path, f'{os.path.basename(pseudo_true_value_path)[:-4]}_{checker_type}.json')
    os.makedirs(checker_save_path, exist_ok=True)

    attr = {
        'result': pseudo_no_type_recharge_rate >= 0.9,
        'is_checked': True,
        'rate': pseudo_no_type_recharge_rate,
        'other': {
            'clip_name': os.path.basename(pseudo_true_value_path)[:-4],
            'checker_type': checker_type,
            "start_end_frame": start_end_frame,
            'gt_id': dominant_id_list,
            '回灌版本与伪真值做比cipv通过率': pseudo_no_type_recharge_rate,
            '回灌版本与伪真值做比cipv不同次数': pseudo_no_type_recharge,
            '回灌版本与伪真值做比cipv不同帧数': pseudo_no_type_recharge_list,
        }
    }

    with open(save_file_path, 'w', encoding='utf-8') as f:
        json.dump(attr, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="运行 cipv跳变 checker 脚本")
    parser.add_argument("pseudo_true_value_path", help="伪真值结果路径")
    parser.add_argument("recharge_res_path", help="回灌结果路径")
    parser.add_argument("checker_save_path", help="checker结果保存路径")
    args = parser.parse_args()

    pseudo_true_value_path = args.pseudo_true_value_path
    recharge_res_path = args.recharge_res_path
    checker_save_path = args.checker_save_path
    checker_type = 'cipv跳变'

    save_file_path = os.path.join(checker_save_path, f'{os.path.basename(pseudo_true_value_path)[:-4]}_{checker_type}.json')
    os.makedirs(checker_save_path, exist_ok=True)

    try:
        checker(pseudo_true_value_path, recharge_res_path, checker_save_path, checker_type)
    except Exception as e:
        error_msg = str(e)
        print(f"[ERROR] Checker failed: {error_msg}")
        attr = {
            'result': False,
            'is_checked': True,
            'rate': 0.0,
            'other': {
                'clip_name': os.path.basename(pseudo_true_value_path)[:-4],
                'checker_type': checker_type,
                'error': error_msg
            }
        }
        with open(save_file_path, 'w', encoding='utf-8') as f:
            json.dump(attr, f, ensure_ascii=False, indent=4)