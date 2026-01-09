import os
import json
import argparse
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean



def align_trajectories(traj1, traj2):
    """
    对齐轨迹，仅保留 traj1 和 traj2 共有的 frame_id，并按 traj2 的 frame_id 范围排序
    """
    # 提取 traj2 的 frame_id 集合
    traj2_frames = {list(point.keys())[0] for point in traj2}

    # 从 traj1 中筛选出与 traj2 共有的 frame_id
    aligned_traj1 = []
    for point in traj1:
        frame_id = list(point.keys())[0]
        if frame_id in traj2_frames:
            aligned_traj1.append(point)

    # 按 traj2 的 frame_id 顺序排序（假设 traj2 已按时间顺序排列）
    aligned_traj1.sort(key=lambda p: list(p.keys())[0])
    aligned_traj2 = sorted(traj2, key=lambda p: list(p.keys())[0])

    return aligned_traj1, aligned_traj2

def trajectory_similarity(traj1, traj2):
    """
    计算对齐后的轨迹相似性（确保 frame_id 完全对齐）
    """
    if not traj1 or not traj2:
        raise ValueError("输入的轨迹为空")

    # 对齐轨迹
    aligned_traj1, aligned_traj2 = align_trajectories(traj1, traj2)
    if len(aligned_traj1) != len(aligned_traj2):
        raise ValueError("对齐后轨迹长度不一致，请检查数据")

    # 提取数值序列（对齐后）
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

    pos_dist, _ = fastdtw(pos1, pos2, dist=euclidean)  # 用 动态时间规整 (DTW) 算法计算 GT 与 DT 在 位置、速度、尺寸 三个维度上的差异  fastdtw 会对齐两条时间序列，允许有时间偏差  euclidean 表示使用欧几里得距离来计算两帧之间的差异
    vel_dist, _ = fastdtw(vel1, vel2, dist=euclidean)
    size_dist, _ = fastdtw(size1, size2, dist=euclidean)
    n = len(values2)
    normalized_pos = pos_dist / n  # 把 DTW 总距离除以样本帧数，得到 平均误差（归一化误差），避免轨迹长短影响
    normalized_vel = vel_dist / n
    normalized_size = size_dist / n
    # weights = {"pos": 1.0, "vel": 0.6, "size": 0.3}  # 设置不同维度的权重
    weights = {"pos": 1.0, "vel": 0, "size": 0}  # 设置不同维度的权重
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
        dist = trajectory_similarity(gt_traj, algo_traj)
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
    if isinstance(signal_type, tuple):
        return best_match_file_data.get(signal_type[0]), best_match_file_data.get(signal_type[1])
    elif isinstance(signal_type, str):
        return best_match_file_data.get(signal_type)
    if isinstance(signal_type, tuple):
        return None, None
    elif isinstance(signal_type, str):
        return None


def check_met_threshold(pseudo_truth_dict, best_match_dict, checker_type,pseudo_attribute):
    prev_id = None
    none_count = 0
    id_jump_records = []
    fail_num = 0
    omissions_num = 0
    fail_frame_value_dict = {}

    for file_name in pseudo_truth_dict:
        best_match_file_data = best_match_dict.get(file_name)

        if best_match_file_data is None:
            omissions_num += 1
            fail_num += 1
            fail_frame_value_dict[file_name] = '目标漏检'
            none_count += 1
            continue
        else:
            current_id = best_match_file_data.get('u8Id')
            if prev_id is not None and prev_id != current_id and none_count <= 3:
                id_jump_records.append((file_name, prev_id, current_id))
            prev_id = current_id
            none_count = 0
        pseudo_truth_value = pseudo_truth_dict[file_name][pseudo_attribute]
        recharge_value = get_recharge_value(best_match_file_data, checker_type)
        if recharge_value is None:
            fail_num += 1
            fail_frame_value_dict[file_name] = '目标漏检'
        elif abs(recharge_value - pseudo_truth_value) > 5:
            fail_num += 1
            fail_frame_value_dict[file_name] = abs(recharge_value - pseudo_truth_value)
        print(
            f'frameid:{file_name} objID:{prev_id} Preception:{recharge_value} GT:{pseudo_truth_value} Diff:{abs(recharge_value - pseudo_truth_value)} result:{False if abs(recharge_value - pseudo_truth_value) > 5 else True}')

    fail_rate = float(fail_num / len(pseudo_truth_dict))
    if fail_rate <= 0.1:
        conclusion = True
    else:
        conclusion = False

    missing_rate = omissions_num / len(pseudo_truth_dict)
    return fail_rate, fail_frame_value_dict, conclusion, missing_rate, id_jump_records




def gt_extract_features(obj):
    return [
        obj['longDistance'],
        obj['latDistance'],
        obj['absolute_longitudinal_velocity'],
        obj['absolute_lateral_velocity'],
        obj['length'],
        obj['width'],
        obj['height'],
    ] if 'longDistance' in obj else []


def dt_extract_features(obj):
    return [
        obj['f32LongDistance'],
        obj['f32LatDistance'],
        obj['f32AbsoluteLongVelocity'],
        obj['f32AbsoluteLatVelocity'],
        obj['f32Length'],
        obj['f32Width'],
        obj['f32Height'],
    ] if 'f32LongDistance' in obj else []


def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        file_datas = [json.loads(file_data.strip()) for file_data in f]
    return file_datas


def checker(pseudo_true_value_path,recharge_res_path,checker_save_path,checker_type):
    pseudo_true_values = read_file(pseudo_true_value_path)
    start_end_frame = pseudo_true_values[0]["frame_id"], pseudo_true_values[-1]["frame_id"]
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
    dt_id_value_dict = {}  # {id:[帧数：数据(横纵向测距测速长宽高)]}
    gt_id_value_dict = {}  # {id:[帧数：数据(横纵向测距测速长宽高)]}
    dt_objs_dict = {}  # {id:[帧数：数据(帧数据)]}
    gt_objs_dict = {}  # {帧数：数据(帧数据)}
    for pseudo_true_value in pseudo_true_values:
        frame = pseudo_true_value["frame_id"]
        pseudo_true_value.update(pseudo_true_value["generated_objects"][0])
        gt_objs_dict[frame] = pseudo_true_value
        gt_id = pseudo_true_value['ID']
        gt_value = gt_extract_features(pseudo_true_value)
        if gt_id not in gt_id_value_dict:
            gt_id_value_dict[gt_id] = []
        gt_id_value_dict[gt_id].append({frame: gt_value})
        if len(recharge_res[int(frame)]) == 0:
            continue
        recharge_objs = [obj for obj in recharge_res[int(frame)]["Oem"]["visInfo"][0]["obj"]["objAttributes"] if obj["u8OemObjClass"] not in [5]]  # 过滤算法类型为5的三角锥数据
        for recharge_obj in recharge_objs:
            dt_id = recharge_obj['u8Id']
            if dt_id not in dt_objs_dict:
                dt_objs_dict[dt_id] = []
            dt_objs_dict[dt_id].append({frame: recharge_obj})
            dt_value = dt_extract_features(recharge_obj)
            if dt_id not in dt_id_value_dict:
                dt_id_value_dict[dt_id] = []
            dt_id_value_dict[dt_id].append({frame: dt_value})

    best_match = find_best_match(gt_id_value_dict, dt_id_value_dict)
    matched_groups = find_covering_detections(best_match, start_end_frame, min_gap=20)
    matched_groups = sorted(matched_groups, key=lambda x: (x[2], x[3]))  # (id号 距离 起始帧 结束帧)
    print(matched_groups)
    dt_matched_dict = {}
    dominant_id_list = []
    for matched_group in matched_groups:
        id = matched_group[0]
        dominant_id_list.append(id)
        dt_obj_list = dt_objs_dict[id]
        for item in dt_obj_list:
            dt_matched_dict.update(item)

    fail_rate, fail_frame_value_dict, conclusion, miss_rate, id_jump_records = check_met_threshold(gt_objs_dict,dt_matched_dict,checker_type,pseudo_attribute)
    pass_rate = round(1-fail_rate, 2)
    save_file_path = os.path.join(checker_save_path,f'{os.path.basename(pseudo_true_value_path)[:-4]}_{checker_type}.json')
    is_checked = True
    attr = {
        'result': True if int(pass_rate) == 1 else False,
        'is_checked': is_checked,
        'rate': pass_rate,
        'other': {
            'clip_name': os.path.basename(pseudo_true_value_path)[:-4],
            'checker_type': checker_type,
            "start_end_frame": start_end_frame,
            'gt_id': dominant_id_list,
            '目标相似度得分': [{item[0]: item[1]} for item in matched_groups],
            'fail_frame': fail_frame_value_dict,
            '漏检比例': miss_rate,
            'id跳变': id_jump_records
        } if is_checked else {}
    }

    with open(save_file_path, 'w', encoding='utf-8') as f:
        json.dump(attr, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="运行 checker 脚本")
    parser.add_argument("pseudo_true_value_path", help="伪真值结果路径")
    parser.add_argument("recharge_res_path", help="回灌结果路径")
    parser.add_argument("checker_save_path", help="checker结果保存路径")

    args = parser.parse_args()

    pseudo_true_value_path = args.pseudo_true_value_path
    recharge_res_path = args.recharge_res_path
    checker_save_path = args.checker_save_path

    print("伪真值结果路径:", pseudo_true_value_path)
    print("回灌结果路径:", recharge_res_path)
    print("checker结果保存路径:", checker_save_path)
    checker_type = '纵向距离跳变'
    try:
        checker(pseudo_true_value_path, recharge_res_path, checker_save_path, checker_type)
    except:
        attr = {
            'result': False,
            'is_checked': True,
            'rate': 0,
            'other': {
                'clip_name': os.path.basename(pseudo_true_value_path)[:-4],
                'checker_type': checker_type,
                'matched_groups': '伪真值目标与回灌版本目标出现错误 请结合原始数据与代码分析'
            }
        }
        save_file_path = os.path.join(checker_save_path,f'{os.path.basename(pseudo_true_value_path)[:-4]}_{checker_type}.json')

    # # 伪真值结果路径
    # pseudo_true_value_path = r'\\hz-iotfs02\Model_Test\TestSpace\Personal_Space\BadCases\VisInsight_20250821093802-2115_2120-漏检.txt'
    # # 回灌结果路径
    # recharge_res_path = r'\\hz-iotfs02\Model_Test\TestSpace\Personal_Space\BadCases\VisInsight_20250821093802-回灌结果.txt'
    # # checker结果保存路径
    # checker_save_path = r'D:\Software_install\Microsoft_Edge\Download\linshi'
    # # checker类型
    # checker_type = '纵向测速跳变'
    # checker(pseudo_true_value_path, recharge_res_path, checker_save_path, checker_type)
