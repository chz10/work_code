import os
import math
import json
import argparse
import numpy as np
import pandas as pd


def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        file_datas = [json.loads(file_data.strip()) for file_data in f]
    return file_datas


def checker(pseudo_true_value_path,recharge_res_path,checker_save_path,checker_type):
    pseudo_true_values = read_file(pseudo_true_value_path)
    start_end_frame = pseudo_true_values[0]["frame_id"], pseudo_true_values[-1]["frame_id"]
    recharge_res = read_file(recharge_res_path)
    all_result = []
    for pseudo_true_value in pseudo_true_values:
        pseudo_true_value.update(pseudo_true_value["generated_objects"][0])
        frame = pseudo_true_value["frame_id"]
        recharge_data= [obj for obj in recharge_res[frame]["Oem"]["visInfo"][0]["obj"]["objAttributes"] if obj["u8OemObjClass"] not in [5]]  # 过滤算法类型为5的三角锥数据
        dt_size = len(recharge_data)
        dintance_array = np.zeros((1, dt_size), dtype=float)
        if dintance_array.size != 0:
            for j in range(dt_size):
                dintance_array[0][j] = distance(pseudo_true_value, recharge_data[j])
            dintance_array_copy = dintance_array.copy()
            gt_index_list, dt_index_list, dist = get_min_or_max_index(dintance_array_copy, get_max=False)
            vision_distance = math.sqrt(pseudo_true_value["longDistance"] ** 2 + (3 * pseudo_true_value["latDistance"]) ** 2)
            if dist > max(4, vision_distance * 0.2):
                dist = -1
                all_result.append({'dist': dist, **pseudo_true_value})
            else:
                all_result.append({'dist': dist, **pseudo_true_value, **recharge_data[dt_index_list]})

    df = pd.DataFrame(all_result)
    # 总数和通过数
    total = len(df)
    df['match_pass'] = df['dist'] > 0
    match_pass_sum = df['match_pass'].sum()
    pass_rate = 1 - (match_pass_sum / total) if total > 0 else 0
    fail_rate = 1 - pass_rate
    save_file_path = os.path.join(checker_save_path,f'{os.path.basename(pseudo_true_value_path)[:-4]}_{checker_type}.json')
    is_checked = True
    attr = {
        'result': True if int(pass_rate) == 1 else False,
        'is_checked': True,
        'rate': round(pass_rate, 2),
        'other': {
            'clip_name': os.path.basename(pseudo_true_value_path)[:-4],
            'checker_type': checker_type,
            "total_frame": total,
            "start_end_frame": start_end_frame,
            "pass_rate": round(pass_rate, 2),
            "pass_id": [],
            "pass_frame": df.loc[~df['match_pass'], 'frame_id'].tolist(),
            "fail_rate": round(fail_rate, 2),
            "fail_id": df.loc[df['match_pass'], 'u8Id'].tolist() if fail_rate > 0 else [],
            "fail_frame": df.loc[df['match_pass'], 'frame_id'].tolist() if fail_rate > 0 else [],
        } if is_checked else {}
    }

    with open(save_file_path, 'w', encoding='utf-8') as f:
        json.dump(attr, f, ensure_ascii=False, indent=4)


def distance(gt_data, dt_data):

    x_dist = abs(gt_data["longDistance"] - dt_data["f32LongDistance"])  # 纵向
    y_dist = abs(gt_data["latDistance"] - dt_data["f32LatDistance"])  # 横向

    weighted_distance = math.sqrt(x_dist ** 2 + (3 * y_dist) ** 2)
    # vision_distance = math.sqrt(gt_data["longDistance"] ** 2 + (3 * gt_data["latDistance"]) ** 2)
    # if weighted_distance > max(3, vision_distance * 0.2):
    #     weighted_distance = float('inf')
    return weighted_distance


def get_min_or_max_index(array, get_max=False):
    mark_index_list = []
    detect_index_list = []
    iou_result_list = []
    narry_size = min(array.shape)
    while narry_size != 0:
        if get_max:
            index = np.unravel_index(array.argmax(), array.shape)
            mark_row, detect_row = index
            if array[mark_row][detect_row] == 0.0:
                break
        else:
            index = np.unravel_index(array.argmin(), array.shape)
            mark_row, detect_row = index
            if array[mark_row][detect_row] == float('inf'):
                break
        # 判断索引是否在
        if mark_row not in mark_index_list and detect_row not in detect_index_list:
            mark_index_list.append(mark_row)
            detect_index_list.append(detect_row)
            iou_result_list.append(array[mark_row][detect_row])
            narry_size -= 1
        if get_max:
            array[mark_row][detect_row] = 0.0
        else:
            array[mark_row][detect_row] = float('inf')
    return mark_index_list[0] if mark_index_list else -1, detect_index_list[0] if detect_index_list else -1, iou_result_list[0] if iou_result_list else -1

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
    checker_type = '误检'

    save_file_path = os.path.join(
        checker_save_path,
        f'{os.path.basename(pseudo_true_value_path)[:-4]}_{checker_type}.json'
    )

    try:
        checker(pseudo_true_value_path, recharge_res_path, checker_save_path, checker_type)
    except Exception as e:
        print(f"[ERROR] Checker failed: {e}")
        attr = {
            'result': False,
            'is_checked': True,
            'rate': 0,
            'other': {
                'clip_name': os.path.basename(pseudo_true_value_path)[:-4],
                'checker_type': checker_type,
                'error': str(e),  # 记录具体错误信息
                'matched_groups': '伪真值目标与回灌版本目标出现错误 请结合原始数据与代码分析'
            }
        }
        # 确保目录存在并写入文件
        os.makedirs(checker_save_path, exist_ok=True)
        with open(save_file_path, 'w', encoding='utf-8') as f:
            json.dump(attr, f, ensure_ascii=False, indent=4)

    # # 伪真值结果路径
    # pseudo_true_value_path = r'\\hz-iotfs02\Model_Test\TestSpace\Personal_Space\BadCases\VisInsight_20250821093802-2115_2120-漏检.txt'
    # # 回灌结果路径
    # recharge_res_path = r'\\hz-iotfs02\Model_Test\TestSpace\Personal_Space\BadCases\VisInsight_20250821093802-回灌结果.txt'
    # # checker结果保存路径
    # checker_save_path = r'\\hz-iotfs02\Model_Test\TestSpace\Personal_Space\BadCases'
    # # checker类型
    # checker_type = '误检'
    # checker(pseudo_true_value_path, recharge_res_path, checker_save_path, checker_type)