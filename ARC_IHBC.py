import os
import shutil
import json
import matplotlib.pyplot as plt


def find_specified_files(directory, file_extension=".txt"):
    """获取目录下所有指定后缀的文件"""
    file_dict = {}
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(file_extension):
                filename = os.path.splitext(file)[0]
                file_dict[filename] = os.path.join(root, file)
    return file_dict

def mkdir_or_exist(directory):
    """创建目录，如果目录不存在则创建"""
    if not os.path.exists(directory):
        os.makedirs(directory)


def get_old_speed(path, file, less, more):
    data1 = {}
    file_path = None
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith('Signal.txt'):
                if file in filename:
                    file_path = os.path.join(dirpath, filename)
                    print(f'找到文件：{file_path}')
                    break
    
    if file_path is None:
        print(f'未找到车机信号文件：{file}')
        return

    with open(file_path, 'r', encoding='utf-8') as lines:
        line = lines.readlines()
        for key, line in enumerate(line):
            # if key >= less and key <= more:
                line_json = json.loads(line)
                data = line_json.get(str(key))
                if not data:
                    continue
                signal_data = data.get('carSignal')
                if signal_data:
                    speed_data = signal_data[0].get('speed')
                    if speed_data is not None:

                        if key not in data1:
                            data1[key] = float(round(speed_data, 2))

        return data1


def get_old_longdistance(path, file, less, more):
    data1 = {}
    file_path = None
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith('rt3003.txt'):
                if file in filename:
                    file_path = os.path.join(dirpath, filename)
                    print(f'找到文件：{file_path}')
                    break
    if file_path is None:
        print(f'未找到RT数据文件：{file}')
        return
    
    with open(file_path, 'r', encoding='utf-8') as lines:
        line = lines.readlines()
        for key, line in enumerate(line):
            # if key >= less and key <= more:
                line_json = json.loads(line)
                data = line_json.get(str(key))

                if not data:
                    continue
                for format1 in data:
                    rt_longdis = format1.get('Range1PosForward')
                    if rt_longdis is not None:
                        if key not in data1:
                            data1[key] = float(round(rt_longdis, 2))
    return data1


def get_new_ihbc(root, file, less, more):
    data1 = {}   # frame: hlbDecision
    list1, list2, list3 = [], [], []
    file_path = None

    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if filename.endswith('.txt') and file in filename:
                file_path = os.path.join(dirpath, filename)
                print(f'找到文件：{file_path}')
                break

    with open(file_path, 'r', encoding='utf-8') as lines:
        for line in lines:
            line = line.strip()
            if not line:
                continue
            line_json = json.loads(line)
            oem = line_json.get('Oem')
            if oem is not None:
                frame = oem.get('u64FrameId')
                # if frame is not None and less <= frame <= more:
                crtl_json = line_json.get('CtrlInfo')
                if crtl_json and 'IHLB' in crtl_json:
                    ihbc_state = crtl_json['IHLB'].get('hlbDecision')
                    if ihbc_state is not None:
                        data1[frame] = int(ihbc_state)

    # 检测状态变化帧
    for i in sorted(data1.keys()):
        if data1.get(i) == 0 and data1.get(i + 1) == 1:
            list1.append(i + 1)
        if data1.get(i) == 1 and data1.get(i + 1) == 2:
            list2.append(i + 1)
        if data1.get(i) == 2 and data1.get(i + 1) == 1:
            list3.append(i + 1)

    # ✅ 返回4个值
    return list1, list2, list3, data1



def get_image(speed_data, longdistance_data, ihbc_data, filename):
    """
    speed_data: dict {frame: speed}
    longdistance_data: dict {frame: distance}
    ihbc_data: dict {frame: hlbDecision}
    """

    # 创建图
    fig, axes = plt.subplots(2, 1, figsize=(9, 7))

    # 设置网格
    for ax in axes:
        ax.grid(True, linestyle='--', linewidth=0.5)

    # --------------------------
    # 第一幅图：速度 vs IHBC状态
    # --------------------------
    ax1 = axes[0]
    ax1.plot(speed_data.keys(), speed_data.values(), color='blue', label='Speed (km/h)')
    ax1.set_ylabel('Speed (km/h)', color='blue')

    # 创建右轴显示IHBC状态
    ax1_r = ax1.twinx()
    ax1_r.plot(ihbc_data.keys(), ihbc_data.values(), color='orange', linestyle='-', drawstyle='steps-post', label='IHBC State')
    ax1_r.set_ylabel('IHBC State', color='orange')
    ax1_r.set_yticks([0, 1, 2])
    ax1_r.set_yticklabels(['OFF(0)', 'HIGH(1)', 'LOW(2)'])
    ax1.set_title(f'{filename} Speed & IHBC State')

    # 添加图例
    ax1.legend(loc='upper left')
    ax1_r.legend(loc='upper right')

    # --------------------------
    # 第二幅图：前向距离 vs IHBC状态
    # --------------------------
    ax2 = axes[1]
    ax2.plot(longdistance_data.keys(), longdistance_data.values(), color='green', label='RT Range (m)')
    ax2.set_ylabel('Range (m)', color='green')

    # 右轴绘制IHBC状态
    ax2_r = ax2.twinx()
    ax2_r.plot(ihbc_data.keys(), ihbc_data.values(), color='orange', linestyle='-', drawstyle='steps-post', label='IHBC State')
    ax2_r.set_ylabel('IHBC State', color='orange')
    ax2_r.set_yticks([0, 1, 2])
    ax2_r.set_yticklabels(['OFF(0)', 'HIGH(1)', 'LOW(2)'])
    ax2.set_title('Range & IHBC State')

    ax2.legend(loc='upper left')
    ax2_r.legend(loc='upper right')

    plt.tight_layout()
    plt.show()
    plt.pause(1)
    plt.close()



if __name__ == "__main__":
    # old_vision = r"\\hz-iotfs02\Function_Test\Front_Camera\#24658_FT_ADAS\202510_Anji_NO2_AEB_IHBC_data\IHBC"
    # new_vision = r'\\hz-iotfs02\Model_Test\TestSpace\Personal_Space\ZYF3\SouthLake\adas_perception_v3.1_SPC030_2m\output\output'

    old_vision = r"\\Yj15155-02161\ihbc"
    new_vision = r'\\hz-iotfs02\Model_Test\TestSpace\Personal_Space\ZYF3\SouthLake\adas_perception_v3.1_SPC030_2m\output_test\buzongtao\20251111_V3.1.27223.1272_2M\lixiang1'

    filename = r'20251023190236'
    start_frame = 0 
    end_frame = 2373

    recharge_path = find_specified_files(new_vision, '.txt')
    for filename, file_path in recharge_path.items():
        print("正在处理文件：", filename)
        
        start_frame = 0
        end_frame = 978

        data1 = get_old_speed(old_vision, filename, start_frame, end_frame)
        data2 = get_old_longdistance(old_vision, filename, start_frame, end_frame)

        if data1 is None:
            print(f"{filename}中未找到车机，跳过此文件")
            continue
        # 现在 get_new_ihbc 返回4个值
        list1, list2, list3, ihbc_data = get_new_ihbc(new_vision, filename, start_frame, end_frame)

        # ✅ 传入 ihbc_data
        get_image(data1, data2, ihbc_data, filename)