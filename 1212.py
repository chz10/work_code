# 处理指定行（例如第二行）
target_line = 2  

if line_num == target_line:
    try:
        data = json.loads(line)
        
        filename = data.get('filename', '')
        if not filename:
            print(f"警告: 在文件 {file_path} 的第 {line_num} 行未找到 'filename' 字段。")
            outfile.write(line + '\n')
            continue
        
        old_prefix = '//192.168.170.5/ly7366/download/827359/494457'
        new_prefix = '//Material/xuekangkang/download/857249/494457'
        
        refilename = None
        if old_prefix in filename and new_prefix not in filename:
            refilename = filename.replace(old_prefix, new_prefix)

        # ⬇⬇⬇ 打印修改前后对比 ⬇⬇⬇
        print(f"【第 {line_num} 行处理】")
        print(f"  原 filename: {filename}")
        print(f"  新 filename: {refilename if refilename else filename}")

        # 写回 JSON
        if refilename is not None:
            data['filename'] = refilename
        else:
            log_file.write(f"{file_path}\n")

        outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
        continue    # 处理完第二行跳过后续逻辑
    
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: 文件 {file_path} 的第 {line_num} 行。错误详情: {e}")
        outfile.write(line + '\n')
        continue

# ↓ 其他行保持原样
outfile.write(line + '\n')
