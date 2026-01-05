import os

def compare_files(txt_file_path, target_directory):
    # 读取txt文件中的文件列表
    with open(txt_file_path, 'r', encoding='utf-8') as f:
        txt_files = {line.strip() for line in f if line.strip()}
    
    # 获取目录中所有的txt文件
    dir_files = {f for f in os.listdir(target_directory) if f.endswith('.txt')}
    
    # 找出差异
    missing_in_dir = txt_files - dir_files  # txt中有但目录中没有的文件
    extra_in_dir = dir_files - txt_files    # 目录中有但txt中没有的文件
    
    return missing_in_dir, extra_in_dir

def main():
    # 设置路径
    txt_file_path = r"\\hz-iotfs02\Model_Test\TestSpace\Personal_Space\ZYF1\SouthLake\adas_perception_v3.1_SPC030_2m\output\FYQ\DTC_lixiang2\0923\output\未替换成功的txt的文件.txt"
    target_directory = r"\\hz-iotfs02\Model_Test\TestSpace\Personal_Space\ZYF1\SouthLake\adas_perception_v3.1_SPC030_2m\output\FYQ\DTC_lixiang2\0923\output"
    
    # 检查路径是否存在
    if not os.path.exists(txt_file_path):
        print(f"错误：txt文件 {txt_file_path} 不存在！")
        return
    if not os.path.exists(target_directory):
        print(f"错误：目标目录 {target_directory} 不存在！")
        return
    
    # 比较文件
    missing_in_dir, extra_in_dir = compare_files(txt_file_path, target_directory)
    
    # 输出结果
    print("\n=== 文件对比结果 ===")
    
    if missing_in_dir:
        print(f"\n在txt文件中存在但目录中不存在的文件（共{len(missing_in_dir)}个）：")
        for file in sorted(missing_in_dir):
            print(f"- {file}")
    else:
        print("\n没有在txt文件中存在但目录中不存在的文件")
    
    if extra_in_dir:
        print(f"\n在目录中存在但txt文件中不存在的文件（共{len(extra_in_dir)}个）：")
        for file in sorted(extra_in_dir):
            print(f"- {file}")
    else:
        print("\n没有在目录中存在但txt文件中不存在的文件")
    
    # 保存结果到文件
    output_file = os.path.join(target_directory, "文件对比结果.txt")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=== 文件对比结果 ===\n\n")
        
        if missing_in_dir:
            f.write(f"在txt文件中存在但目录中不存在的文件（共{len(missing_in_dir)}个）：\n")
            for file in sorted(missing_in_dir):
                f.write(f"- {file}\n")
        else:
            f.write("没有在txt文件中存在但目录中不存在的文件\n")
        
        f.write("\n")
        
        if extra_in_dir:
            f.write(f"在目录中存在但txt文件中不存在的文件（共{len(extra_in_dir)}个）：\n")
            for file in sorted(extra_in_dir):
                f.write(f"- {file}\n")
        else:
            f.write("没有在目录中存在但txt文件中不存在的文件\n")
    
    print(f"\n结果已保存到：{output_file}")

if __name__ == "__main__":
    main()
