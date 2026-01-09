import os
import json
from typing import List
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import runner


# -------------------------工具函数-------------------------
def get_files_path(folder_path: str, extensions: List[str], recursive: bool = True) -> List[str]:
    """
    获取指定目录下所有指定后缀文件的路径列表

    :param folder_path: 根目录路径
    :param extensions: 文件后缀列表，如 ['.txt', '.json']
    :param recursive: 是否递归子目录
    :return: 文件完整路径列表
    """
    extensions = [ext.lower() for ext in extensions]
    matched_files = []

    if recursive:
        for root, dirs, files in os.walk(folder_path):
            for f in files:
                if any(f.lower().endswith(ext) for ext in extensions):
                    matched_files.append(os.path.join(root, f))
    else:
        for f in os.listdir(folder_path):
            full_path = os.path.join(folder_path, f)
            if os.path.isfile(full_path) and any(f.lower().endswith(ext) for ext in extensions):
                matched_files.append(full_path)

    return matched_files


def get_pseudo_true_files_map(gt_root_path, match_name_list, need_to_matched_checker_list):
    """
    根据 GT 文件夹和 checker 类型列表，生成 clip_name -> pseudo_true 文件列表的映射

    :param gt_root_path: GT 文件夹路径
    :param match_name_list: 需要匹配的 clip 视频名称列表
    :param need_to_matched_checker_list: 需要匹配的 checker 名称列表
    :return: {clip_name: [pseudo_true_file1, pseudo_true_file2, ...]} 字典
    """
    pseudo_true_files_map = {}
    checker_type_list = [checker_type.replace('.py', '').replace('南湖-', '').replace('_checker', '')
                         for checker_type in need_to_matched_checker_list]

    for root, dirs, filenames in os.walk(gt_root_path):
        for filename in filenames:
            if not filename.endswith('.txt'):
                continue
            filename_parts = filename.split('-')
            if len(filename_parts) >= 2:
                checker_type = filename_parts[-1].replace('.txt', '')
                clip_video_name = filename_parts[0]
                if clip_video_name in match_name_list and checker_type in checker_type_list:
                    if pseudo_true_files_map.get(clip_video_name) is None:
                        pseudo_true_files_map[clip_video_name] = []
                    pseudo_true_files_map[clip_video_name].append(os.path.join(root, filename))
    return pseudo_true_files_map


def get_checker_path_list(checker_dir, need_to_matched_checker_list):
    """
    获取 checker 脚本的完整路径列表

    :param checker_dir: checker 脚本目录
    :param need_to_matched_checker_list: 需要匹配的 checker 名称列表
    :return: checker 脚本路径列表
    """
    path_list = []
    all_checker_path_list = get_files_path(checker_dir, ['.py'])
    for path in all_checker_path_list:
        name = os.path.basename(path).replace('.py', '')
        if name in need_to_matched_checker_list:
            path_list.append(path)
    return path_list


def generate_checker_result_table(input_checker_result_dir, algo_version):
    """
    生成所有 checker 的结果汇总表格

    :param input_checker_result_dir: checker 输出 JSON 文件目录
    :param algo_version: 算法版本
    :return: pandas DataFrame
    """
    results = []
    for root, dirs, filenames in os.walk(input_checker_result_dir):
        for filename in filenames:
            if filename.endswith('.json'):
                filepath = os.path.join(root, filename)
                try:
                    content = open(filepath, 'r', encoding='utf-8').read()
                    checker_result = json.loads(content)
                    result_item = {
                        'clip_name': checker_result['other']['clip_name'],
                        'checker_type': checker_result['other']['checker_type'],
                        'algo_version': algo_version,
                        'result': checker_result['result'],
                        'is_checked': checker_result['is_checked'],
                        'rate': checker_result['rate']
                    }
                    results.append(result_item)
                except Exception as e:
                    print(f'Error generate_checker_result_table: {e}')
    df = pd.DataFrame(results)
    return df


# -------------------------多线程执行函数-------------------------
def run_checker_task(checker_path, pseudo_true_file_path, recharge_res_path, checker_save_path, log_dir):
    """
    线程池任务：调用 CheckerRunner 执行单个 checker

    :param checker_path: checker 脚本路径
    :param pseudo_true_file_path: pseudo_true 文件路径
    :param recharge_res_path: 回灌结果路径
    :param checker_save_path: checker 输出保存目录
    :param log_dir: 日志输出目录
    :return: (checker_path, pseudo_true_file_path, retcode)
    """
    runner_instance = runner.CheckerRunner(checker_path)
    retcode = runner_instance.run_as_script(
        pseudo_true_file_path,
        recharge_res_path,
        checker_save_path,
        log_dir
    )
    return (checker_path, pseudo_true_file_path, retcode)


# -------------------------主程序-------------------------
if __name__ == '__main__':
    # -------------------------路径配置-------------------------
    #gt路径
    gt_root_dir = r'E:\Badcase\GT\GT-ALL'
    # checker路径，不用修改此路径
    checker_root_dir = r'E:\Badcase\Checker'
    #回灌后素材路径，不要有真值写入
    algo_result_root_dir = r"\\hz-iotfs02\Model_Test\TestSpace\Personal_Space\ZYF\SouthLake\adas_perception_v3.1_SPC030\output_test\guohao\20251011\lixiang2_V3.0.27223.1200_2M"
    #checker结果路径
    checker_output_root_dir = r"E:\Badcase\checker_results\results\V3.0.27223.1200\20251117"
    #log
    log_output_root_dir = r"E:\Badcase\checker_results\log\V3.0.27223.1200\20251117"
    # checker_list = ['南湖-朝向角跳变_checker']
    checker_list = [
        '南湖-cipv跳变_checker',
        '南湖-朝向角跳变_checker',
        '南湖-横向加速度跳变_checker',
        '南湖-横向距离跳变_checker',
        '南湖-横向速度跳变_checker',
        '南湖-类型跳变_checker',
        '南湖-漏检_checker',
        '南湖-误检_checker',
        '南湖-纵向加速度跳变_checker',
        '南湖-纵向距离跳变_checker',
        '南湖-纵向速度跳变_checker'
    ]
    algo_version = '3.0.2'

    # -------------------------获取文件列表-------------------------
    algo_result_path_list = get_files_path(algo_result_root_dir, ['.txt'])
    algo_result_path_map = {os.path.basename(p).replace('.txt', ''): p for p in algo_result_path_list}
    clip_video_name_list = [os.path.basename(p)[:-4] for p in algo_result_path_list]

    pseudo_true_file_path_list_map = get_pseudo_true_files_map(gt_root_dir, clip_video_name_list, checker_list)
    checker_path_list = get_checker_path_list(checker_root_dir, checker_list)

    # -------------------------多线程执行-------------------------
    cpu_count = os.cpu_count() or 4
    max_workers = max(1, cpu_count)
    print(f"[Runner] CPU 核心数: {cpu_count}, 使用线程数: {max_workers}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {}
        for clip_video_name in clip_video_name_list:
            for checker_path in checker_path_list:
                pseudo_true_file_path_list = pseudo_true_file_path_list_map.get(clip_video_name, [])
                recharge_res_path = algo_result_path_map.get(clip_video_name, None)
                if recharge_res_path is None or len(pseudo_true_file_path_list) == 0:
                    continue

                checker_name = os.path.basename(checker_path).replace('.py', '').replace('南湖-', '').replace('_checker', '')

                for pseudo_true_file_path in pseudo_true_file_path_list:
                    gt_filename = os.path.basename(pseudo_true_file_path)
                    gt_type = gt_filename.rsplit('-', 1)[-1].replace('.txt', '')

                    # checker_type 和 gt_type 完全一致才执行

                    if checker_name != gt_type:
                        continue

                    future = executor.submit(
                        run_checker_task,
                        checker_path,
                        pseudo_true_file_path,
                        recharge_res_path,
                        checker_output_root_dir,
                        log_output_root_dir
                    )
                    future_to_task[future] = (checker_path, pseudo_true_file_path)

        # 输出多线程任务完成情况
        for future in as_completed(future_to_task):
            checker_path, pseudo_true_file_path = future_to_task[future]
            try:
                _, _, retcode = future.result()
                print(f"[完成] {os.path.basename(checker_path)} -> {os.path.basename(pseudo_true_file_path)}, retcode={retcode}")
            except Exception as e:
                print(f"[异常] {os.path.basename(checker_path)} -> {os.path.basename(pseudo_true_file_path)} : {e}")

    # -------------------------生成汇总报表-------------------------
    df = generate_checker_result_table(checker_output_root_dir, algo_version)
    check_result_summary_report_path = os.path.join(checker_output_root_dir, '结果细节统计.xlsx')
    df.to_excel(check_result_summary_report_path, index=False)
    print(f"[完成] 总表已生成: {check_result_summary_report_path}")
