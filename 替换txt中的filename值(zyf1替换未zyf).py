import os
import json

def find_txt_files(directory_path):
    """
    递归查找指定目录及其子目录下的所有 .txt 文件。
    """
    txt_files = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.lower().endswith('.txt'):
                txt_files.append(os.path.join(root, file))
    return txt_files

def extract_h264_name(filename):
    """
    从文件路径中提取h264文件名。
    例如："//路径/1702605577.h264" -> "1702605577.h264"
    """
    return os.path.basename(filename)

def process_txt_file(file_path, output_file):
    """
    处理单个 .txt 文件：仅替换第一行 JSON 数据中的 'filename' 字段，其他行保持不变。
    """
    temp_file_path = file_path + '.tmp'
    replaced = False  # 标志是否已替换第一行

    with open(file_path, 'r', encoding='utf-8') as infile, \
         open(temp_file_path, 'w', encoding='utf-8') as outfile, \
         open(output_file, 'a', encoding='utf-8') as log_file:  # 追加模式打开
        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                outfile.write('\n')
                continue

            # 只处理第一行的 filename
            if not replaced:
                try:
                    data = json.loads(line)
                    filename = data.get('filename', '')
                    if not filename:
                        print(f"警告: 在文件 {file_path} 的第 {line_num} 行未找到 'filename' 字段。")
                        outfile.write(line + '\n')
                        continue
                    
                    refilename = None
                    old_prefix = 'ZYF1/SouthLake/ZYF1/SouthLake'
                    new_prefix = 'ZYF1/SouthLake'
                    if old_prefix in filename:
                        refilename = filename.replace(old_prefix, new_prefix)

                    print('filename', filename, 'refilename', refilename)
                    if refilename is not None:
                        data['filename'] = refilename
                    else:
                        log_file.write(f"{file_path}\n")

                    outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                    replaced = True
                    continue  # 已处理完第一行，跳过后续替换逻辑
                except json.JSONDecodeError as e:
                    print(f"JSON解析错误: 文件 {file_path} 的第 {line_num} 行。错误详情: {e}")
                    outfile.write(line + '\n')
                    continue

            # 后续行不处理，只写回原始内容
            outfile.write(line + '\n')

    os.replace(temp_file_path, file_path)


def main():
    search_path = r"\\hz-iotfs02\Model_Test\TestSpace\Personal_Space\ZYF\SouthLake\adas_perception_v3.1_SPC030_2m\output_test\fulltest\20251202_V3.1.27223.1318_2M\DTC_lixiang1\output"
    output_file_path = r"\\hz-iotfs02\Model_Test\TestSpace\Personal_Space\ZYF\SouthLake\adas_perception_v3.1_SPC030_2m\output_test\fulltest\20251202_V3.1.27223.1318_2M\DTC_lixiang1\output"

    output_file = os.path.join(output_file_path, '未替换成功的txt的文件.txt')

    if os.path.exists(output_file):
        os.remove(output_file)
        # print(f"已删除存在的文件: {output_file}")
    else:
        # print(f"文件 {output_file} 不存在，将创建新文件。")
        pass


    txt_files = find_txt_files(search_path)
    print(f"在 '{search_path}' 下找到 {len(txt_files)} 个 .txt 文件。")

    for txt_file in txt_files:
        print(f"正在处理文件: {txt_file}")
        process_txt_file(txt_file, output_file)

    print("所有文件处理完成。")

if __name__ == "__main__":
    main()
