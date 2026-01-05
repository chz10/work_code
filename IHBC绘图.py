import os
import json
import matplotlib.pyplot as plt


def mkdir_or_exist(directory):
    """创建目录，如果目录不存在则创建"""
    if not os.path.exists(directory):
        os.makedirs(directory)


# ============================================================================================
# 旧版本自车速度
# ============================================================================================
def get_old_speed(path, file, less, more):
    data1 = {}
    file_path = None

    # 查找 *_Signal.txt
    for dirpath, _, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith('Signal.txt') and file in filename:
                file_path = os.path.join(dirpath, filename)
                print(f'找到速度文件：{file_path}')
                break

    if file_path is None:
        print("❌ 未找到速度文件")
        return {}

    with open(file_path, 'r', encoding='utf-8') as lines:
        all_lines = lines.readlines()
        for key, line in enumerate(all_lines):
            if less <= key <= more:
                try:
                    line_json = json.loads(line)
                    data = line_json.get(str(key))
                    if not data:
                        continue
                    signal_data = data.get('carSignal')
                    if signal_data:
                        speed_data = signal_data[0].get('speed')
                        if speed_data is not None:
                            data1[key] = float(round(speed_data, 2))
                except Exception:
                    continue

    return data1


# ============================================================================================
# 旧版本 RT 前向距离
# ============================================================================================
def get_old_longdistance(path, file, less, more):
    data1 = {}
    file_path = None

    # 查找 *_rt3003.txt
    for dirpath, _, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith('rt3003.txt') and file in filename:
                file_path = os.path.join(dirpath, filename)
                print(f'找到RT文件：{file_path}')
                break

    # 未找到 RT 文件
    if file_path is None:
        print("⚠ 未找到 RT 文件，后续只绘制速度+灯光图")
        return None

    with open(file_path, 'r', encoding='utf-8') as lines:
        all_lines = lines.readlines()

        for key, line in enumerate(all_lines):
            if less <= key <= more:
                try:
                    line_json = json.loads(line)
                    data = line_json.get(str(key))
                    if not data:
                        continue
                    for format1 in data:
                        rt_longdis = format1.get('Range1PosForward')
                        if rt_longdis is not None:
                            data1[key] = float(round(rt_longdis, 2))
                except:
                    continue

    return data1


# ============================================================================================
# 新版本 IHBC 状态
# ============================================================================================
def get_new_ihbc(root, file, less, more):
    data1 = {}   # frame: hlbDecision
    file_path = None

    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if filename.endswith('.txt') and file in filename:
                file_path = os.path.join(dirpath, filename)
                print(f'找到IHBC文件：{file_path}')
                break

    if file_path is None:
        print("❌ 未找到 IHBC 文件")
        return {}

    with open(file_path, 'r', encoding='utf-8') as lines:
        for line in lines:
            line = line.strip()
            if not line:
                continue
            line_json = json.loads(line)

            oem = line_json.get('Oem')
            if oem is not None:
                frame = oem.get('u64FrameId')
                if frame is not None and less <= frame <= more:
                    crtl_json = line_json.get('CtrlInfo')
                    if crtl_json and 'IHLB' in crtl_json:
                        ihbc_state = crtl_json['IHLB'].get('hlbDecision')
                        if ihbc_state is not None:
                            data1[frame] = int(ihbc_state)

    return data1


# ============================================================================================
# 绘图 - 情况1：仅速度 + 灯光
# ============================================================================================
def plot_speed_ihbc(speed_data, ihbc_data):
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.grid(True, linestyle='--', linewidth=0.5)

    # 车速
    ax1.plot(speed_data.keys(), speed_data.values(), label='Speed (km/h)')
    ax1.set_ylabel("Speed (km/h)")

    # 灯光状态
    ax2 = ax1.twinx()
    ax2.plot(ihbc_data.keys(), ihbc_data.values(), drawstyle='steps-post', label='IHLB State')
    ax2.set_ylabel("IHBC State")
    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels(['OFF(0)', 'HIGH(1)', 'LOW(2)'])

    plt.title("Speed & IHBC State (No RT File)")
    plt.tight_layout()
    plt.show()


# ============================================================================================
# 绘图 - 情况2：速度 + 灯光 ；距离 + 灯光
# ============================================================================================
def plot_speed_range_ihbc(speed_data, longdistance_data, ihbc_data):
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))

    for ax in axes:
        ax.grid(True, linestyle='--', linewidth=0.5)

    # 图 1：速度 + IHBC
    ax1 = axes[0]
    ax1.plot(speed_data.keys(), speed_data.values(), color='blue')
    ax1.set_ylabel("Speed (km/h)")

    ax1_r = ax1.twinx()
    ax1_r.plot(ihbc_data.keys(), ihbc_data.values(),
               color='orange', drawstyle='steps-post')
    ax1_r.set_ylabel("IHBC State")
    ax1_r.set_yticks([0, 1, 2])
    ax1_r.set_yticklabels(['OFF(0)', 'HIGH(1)', 'LOW(2)'])
    ax1.set_title("Speed & IHBC State")

    # 图 2：RT前向距离 + IHBC
    ax2 = axes[1]
    ax2.plot(longdistance_data.keys(), longdistance_data.values(), color='green')
    ax2.set_ylabel("Range (m)")

    ax2_r = ax2.twinx()
    ax2_r.plot(ihbc_data.keys(), ihbc_data.values(),
               color='orange', drawstyle='steps-post')
    ax2_r.set_ylabel("IHBC State")
    ax2_r.set_yticks([0, 1, 2])
    ax2_r.set_yticklabels(['OFF(0)', 'HIGH(1)', 'LOW(2)'])
    ax2.set_title("Range & IHBC State")

    plt.tight_layout()
    plt.show()


# ============================================================================================
# 主流程
# ============================================================================================
if __name__ == "__main__":

    old_vision = r"\\hz-iotfs02\Model_Test\TestSpace\Personal_Space\DWZ\测试素材\IHBC\测试数据"
    new_vision = r'\\hz-iotfs02\Model_Test\TestSpace\Personal_Space\YJ\southlake\adas_perception_v3.1_SPC030_2m\output\duwenzhe\20251204_3.1.27223.1296_IHBC30'

    filename = r'20251022221750'
    start_frame = 100
    end_frame = 1279

    # 旧速度
    data1 = get_old_speed(old_vision, filename, start_frame, end_frame)

    # RT 距离（可能不存在）
    data2 = get_old_longdistance(old_vision, filename, start_frame, end_frame)

    # IHBC
    ihbc_data = get_new_ihbc(new_vision, filename, start_frame, end_frame)

    # =====================================================================================
    # ⭐ 根据 RT 是否存在自动决定绘图逻辑
    # =====================================================================================
    if data2 is None:
        print("⚠ 未找到 RT 文件，仅绘制速度 + 灯光信号图")
        plot_speed_ihbc(data1, ihbc_data)
    else:
        plot_speed_range_ihbc(data1, data2, ihbc_data)