import os
import sys
import json
import math
# import csv
import re
import traceback
from typing import List, Dict, Any, Tuple

import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import matplotlib.patches as patches

matplotlib.rcParams['font.sans-serif'] = ['SimSun']
matplotlib.rcParams['font.family'] = 'SimSun'
matplotlib.rcParams['axes.unicode_minus'] = False

# 指定真实字体文件
matplotlib.font_manager.fontManager.addfont("C:/Windows/Fonts/simsun.ttc")



# ============================================================
#  解析文本中的 JSON（鲁棒）
# ============================================================
def load_json_objects_from_text(text: str) -> List[Dict[str, Any]]:
    text = text.strip()
    if not text:
        return []

    # 尝试直接解析
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, list) else [obj]
    except:
        pass

    # 尝试 JSON lines
    objs = []
    all_line_json = True
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            objs.append(json.loads(ln))
        except:
            all_line_json = False
            break
    if all_line_json and objs:
        return objs

    # 尝试把 }{ 变成 },{
    replaced = re.sub(r"\}\s*\{", "},{", text)
    try:
        arr = json.loads(f"[{replaced}]")
        return arr
    except:
        pass

    # 退化：用括号匹配提取多个 JSON
    results = []
    stack = []
    start = None

    for i, ch in enumerate(text):
        if ch == "{":
            if start is None:
                start = i
            stack.append("{")
        elif ch == "}":
            if stack:
                stack.pop()
                if not stack and start is not None:
                    snippet = text[start:i+1]
                    start = None
                    try:
                        results.append(json.loads(snippet))
                    except:
                        pass

    return results


# ============================================================
#   计算矩形四个角（含 heading 旋转）
# ============================================================
def rect_corners(cx: float, cy: float, length: float, width: float, heading: float):
    hl = length / 2.0
    hw = width / 2.0

    pts = [
        ( hl,  hw),   # front-right
        ( hl, -hw),   # front-left
        (-hl, -hw),   # rear-left
        (-hl,  hw),   # rear-right
    ]

    cos_h = math.cos(heading)
    sin_h = math.sin(heading)

    rot = []
    for dx, dy in pts:
        rx = dx * cos_h - dy * sin_h
        ry = dx * sin_h + dy * cos_h
        rot.append((cx + rx, cy + ry))
    return rot


# ============================================================
#   保存矩形图片
# ============================================================
def save_rect_image(corners, cx, cy, out_path, title=""):
    fig, ax = plt.subplots(figsize=(6, 6))
    poly = patches.Polygon(corners, closed=True, fill=False, linewidth=2)
    ax.add_patch(poly)

    ax.plot(cx, cy, "rx")

    xs = [p[0] for p in corners]
    ys = [p[1] for p in corners]

    dx = max(5, (max(xs) - min(xs)) * 0.6)
    dy = max(5, (max(ys) - min(ys)) * 0.6)

    ax.set_xlim(cx - dx, cx + dx)
    ax.set_ylim(cy - dy, cy + dy)
    ax.set_aspect("equal")
    ax.grid(True)
    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


# ============================================================
#   处理单个文件
# ============================================================
def process_file(file_path, out_dir):
    basename = os.path.basename(file_path)
    name_noext = os.path.splitext(basename)[0]

    images_dir = os.path.join(out_dir, "images")
    # csvs_dir   = os.path.join(out_dir, "csvs")
    os.makedirs(images_dir, exist_ok=True)
    # os.makedirs(csvs_dir, exist_ok=True)

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    objs = load_json_objects_from_text(text)
    total_objs = len(objs)
    print(f"  -> 发现 {total_objs} 条 JSON 数据")

    rows = []

    for idx, entry in enumerate(objs):
        print(f"     处理对象 {idx+1}/{total_objs} ({(idx+1)/total_objs*100:.1f}%)", end="\r")

        try:
            generated_list = []
            frame_id = entry.get("frame_id", "")
            time_stamp = entry.get("time_stamp", "")

            if "generated_objects" in entry:
                generated_list = entry["generated_objects"]
            else:
                # 单独对象也兼容
                if "longDistance" in entry and "latDistance" in entry:
                    generated_list = [entry]

            # 遍历该 JSON 中的全部对象
            for obj in generated_list:
                cx = obj.get("longDistance")
                cy = obj.get("latDistance")
                length = obj.get("length")
                width = obj.get("width")
                heading = obj.get("heading", 0.0)
                oid = obj.get("ID", "")

                if cx is None or cy is None or length is None or width is None:
                    continue

                cx = float(cx)
                cy = float(cy)
                length = float(length)
                width = float(width)
                heading = float(heading)

                corners = rect_corners(cx, cy, length, width, heading)

                img_name = f"{name_noext}__obj{oid}__{idx}.png"
                img_path = os.path.join(images_dir, img_name)
                save_rect_image(corners, cx, cy, img_path,
                                title=f"{basename} | ID={oid} | frame={frame_id}")

                row = {
                    "source_file": basename,
                    "object_index": idx,
                    "frame_id": frame_id,
                    "time_stamp": time_stamp,
                    "ID": oid,
                    "cx": cx,
                    "cy": cy,
                    "length": length,
                    "width": width,
                    "heading": heading,
                    "img_path": img_path,
                    "corners": corners
                }
                rows.append(row)

        except Exception:
            print(f"\n[ERROR] 处理文件 {basename} 的对象 {idx} 出错")
            traceback.print_exc()

    print()  # 换行

    # # 写 CSV
    # if rows:
    #     csv_path = os.path.join(csvs_dir, f"{name_noext}.csv")
    #     with open(csv_path, "w", newline="", encoding="utf-8") as f:
    #         writer = csv.writer(f)
    #         writer.writerow([
    #             "source_file", "object_index", "frame_id", "time_stamp",
    #             "ID", "cx", "cy", "length", "width", "heading",
    #             "corner0_x", "corner0_y", "corner1_x", "corner1_y",
    #             "corner2_x", "corner2_y", "corner3_x", "corner3_y",
    #             "img_path"
    #         ])

    #         for r in rows:
    #             cs = r["corners"]
    #             writer.writerow([
    #                 r["source_file"], r["object_index"], r["frame_id"], r["time_stamp"],
    #                 r["ID"], r["cx"], r["cy"], r["length"], r["width"], r["heading"],
    #                 cs[0][0], cs[0][1], cs[1][0], cs[1][1],
    #                 cs[2][0], cs[2][1], cs[3][0], cs[3][1],
    #                 r["img_path"]
    #             ])

    # return rows


# ============================================================
#   主函数 — 带文件进度条
# ============================================================
def main():
    input_dir  = r"C:\Users\chz62985\Desktop\CHecker-test\tubiao\123"
    output_dir = r"C:\Users\chz62985\Desktop\CHecker-test\输出"

    all_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(".txt")])
    total_files = len(all_files)

    print(f"\n=== 共发现 {total_files} 个 TXT 文件 ===\n")

    # all_rows = []
    # for idx, fname in enumerate(all_files, start=1):
    #     print(f"\n===== 处理文件 {idx}/{total_files}: {fname} =====")
    #     full_path = os.path.join(input_dir, fname)

    #     rows = process_file(full_path, output_dir)
    #     all_rows.extend(rows)

    # 写主 CSV
    # if all_rows:
    #     csv_path = os.path.join(output_dir, "all_objects.csv")
    #     with open(csv_path, "w", newline="", encoding="utf-8") as f:
    #         writer = csv.writer(f)
    #         writer.writerow([
    #             "source_file", "object_index", "frame_id", "time_stamp",
    #             "ID", "cx", "cy", "length", "width", "heading",
    #             "corner0_x", "corner0_y", "corner1_x", "corner1_y",
    #             "corner2_x", "corner2_y", "corner3_x", "corner3_y",
    #             "img_path"
    #         ])
    #         for r in all_rows:
    #             cs = r["corners"]
    #             writer.writerow([
    #                 r["source_file"], r["object_index"], r["frame_id"], r["time_stamp"],
    #                 r["ID"], r["cx"], r["cy"], r["length"], r["width"], r["heading"],
    #                 cs[0][0], cs[0][1], cs[1][0], cs[1][1],
    #                 cs[2][0], cs[2][1], cs[3][0], cs[3][1],
    #                 r["img_path"]
    #             ])

    #     print(f"\n>>> 处理完成！生成汇总文件： {csv_path}")


if __name__ == "__main__":
    main()

