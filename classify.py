import os

def classify_car_type_from_paths(input_txt_path, output_dir="."):
    """
    从输入txt文件中读取路径，按车型分类并保存到对应txt文件
    :param input_txt_path: 原始路径列表的txt文件路径
    :param output_dir: 输出分类txt文件的保存目录（默认当前目录）
    """
    # 1. 初始化字典，用于存储「车型: 路径列表」的映射关系
    car_type_path_dict = {}

    # 2. 读取原始txt文件中的所有路径
    try:
        with open(input_txt_path, 'r', encoding='utf-8') as f:
            # 读取所有行，去除空行和首尾空白字符
            paths = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"错误：未找到输入文件 {input_txt_path}")
        return
    except Exception as e:
        print(f"错误：读取输入文件失败，原因：{e}")
        return

    # 3. 遍历每个路径，提取车型并分类
    for path in paths:
        # 统一路径分隔符（兼容Windows反斜杠和Linux正斜杠）
        normalized_path = path.replace("\\", "/")
        # 按分隔符拆分路径为各个层级
        path_parts = normalized_path.split("/")
        # 过滤空字符串（避免路径首尾分隔符导致的空元素）
        path_parts = [part for part in path_parts if part]

        # 核心：提取车型名称（关键逻辑，兼容不同路径层级）
        # 分析路径特征：车型是「video目录的上上层目录的父目录」（即video的祖父目录的前一级）
        # 先找到video目录的索引位置
        car_type = None
        try:
            # 找到"video"层级的索引
            video_index = path_parts.index("video")
            # 车型所在层级：video索引往前推3位（兼容绝大多数场景，可灵活调整）
            # 解释：path -> ... / 车型 / 日期目录 / xin_data目录 / video / 文件名
            # video_index - 3 即为车型所在层级（若层级偏差，可微调该数值）
            if video_index >= 3:
                car_type = path_parts[video_index - 3]
            else:
                # 若video层级过前，取video往前推2位（兜底兼容特殊路径）
                car_type = path_parts[video_index - 2] if video_index >= 2 else "未知车型"
        except ValueError:
            # 若路径中无video目录，取倒数第5位（兜底方案，可根据实际情况调整）
            if len(path_parts) >= 5:
                car_type = path_parts[-5]
            else:
                car_type = f"未知车型_{len(car_type_path_dict) + 1}"
        except Exception as e:
            # 极端异常场景，标记为未知车型
            car_type = f"未知车型_{len(car_type_path_dict) + 1}"

        # 4. 将当前路径添加到对应车型的列表中
        if car_type not in car_type_path_dict:
            car_type_path_dict[car_type] = []
        car_type_path_dict[car_type].append(path)

    # 5. 创建输出目录（若不存在）
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 6. 遍历车型字典，保存到对应txt文件
    for car_type, path_list in car_type_path_dict.items():
        # 构造输出文件路径
        output_txt_path = os.path.join(output_dir, f"{car_type}.txt")
        # 写入路径列表
        try:
            with open(output_txt_path, 'w', encoding='utf-8') as f:
                # 每行写入一个路径
                for path in path_list:
                    f.write(f"{path}\n")
            print(f"成功：车型「{car_type}」共 {len(path_list)} 条路径，已保存到 {output_txt_path}")
        except Exception as e:
            print(f"错误：保存车型「{car_type}」的文件失败，原因：{e}")

# ===================== 调用示例（修改此处路径即可使用） =====================
# ===================== 调用示例（修改此处路径即可使用） =====================
if __name__ == "__main__":
    # 方案1：原始字符串（推荐，只需加 r）
    input_txt_file = r"C:\Users\chz62985\Desktop\xkk_test.txt"
    # 可选：修改为你想要保存输出文件的目录，同样可使用 r 避免转义问题
    output_directory = r"C:\Users\chz62985\Desktop\xkk_test3.txt"  # 示例输出目录

    # 执行分类函数
    classify_car_type_from_paths(input_txt_file, output_directory)