import os

def remove_fields_and_duplicates(file_path, fields_to_remove):
    """
    从txt文件中删除指定的字段和重复行
    
    Args:
        file_path: txt文件路径
        fields_to_remove: 要删除的字段列表
    """
    try:
        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 处理每一行
        processed_lines = []
        seen_lines = set()  # 用于记录已处理的行
        
        for line in lines:
            current_line = line.strip()  # 去除首尾空白
            if not current_line:  # 跳过空行
                continue
                
            # 删除指定的字段
            for field in fields_to_remove:
                current_line = current_line.replace(field, '')
            
            # 如果这行还没有出现过，则保留
            if current_line not in seen_lines:
                seen_lines.add(current_line)
                processed_lines.append(current_line + '\n')
        
        # 将处理后的内容写回文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(processed_lines)
            
        print(f"成功处理文件: {file_path}")
        print(f"删除了 {len(lines) - len(processed_lines)} 行（包括重复行和空行）")
        
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {str(e)}")

def main():
    # 设置文件路径
    txt_file_path = r"C:\Users\chz62985\Desktop\素材路径.txt"
    
    # 设置要删除的字段列表
    fields_to_remove = [
        "缺少对应txt文件: ",
        # 在这里添加更多要删除的字段
    ]
    
    # 检查文件是否存在
    if not os.path.exists(txt_file_path):
        print(f"错误：文件 {txt_file_path} 不存在！")
        return
    
    print("开始处理文件...")
    remove_fields_and_duplicates(txt_file_path, fields_to_remove)
    print("处理完成！")

if __name__ == "__main__":
    main()
