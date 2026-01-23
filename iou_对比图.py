import os
import json
import matplotlib.pyplot as plt


# =========================
# 在目录中查找 txt
# =========================
def find_txt(root, filename):
    target = filename + ".txt"
    for r, _, files in os.walk(root):
        if target in files:
            return os.path.join(r, target)
    return None


# =========================
# 读取 jsonl
# =========================
def load_jsonl(path):
    frames = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            try:
                j = json.loads(line)
            except:
                continue

            fid = j.get("frameId") or j.get("frameid")
            if fid is not None:
                frames[int(fid)] = j
    return frames


# =========================
# 一帧：所有 rcBox 总面积
# =========================
def calc_frame_total_area(j):
    total = 0.0
    cnt = 0

    try:
        objs = j["Oem"]["visInfo"][0]["obj"]["objAttributes"]
    except:
        return None, 0

    for o in objs:
        rc = o.get("rcBox")
        if not rc:
            continue

        w = rc["right"] - rc["left"]
        h = rc["bottom"] - rc["top"]
        if w > 0 and h > 0:
            total += w * h
            cnt += 1

    if cnt == 0:
        return None, 0
    return total, cnt


# =========================
# 一帧：指定 ID 的 rcBox 面积
# =========================
def calc_frame_area_by_id(j, target_id):
    try:
        objs = j["Oem"]["visInfo"][0]["obj"]["objAttributes"]
    except:
        return None

    for o in objs:
        obj_id = o.get("u8Id") or o.get("s32ObjID")
        if obj_id != target_id:
            continue

        rc = o.get("rcBox")
        if not rc:
            return None

        w = rc["right"] - rc["left"]
        h = rc["bottom"] - rc["top"]
        if w > 0 and h > 0:
            return w * h

    return None


# =========================
# 主流程
# =========================
def plot_rcbox_area(recharge_root, filename, start=None, end=None, obj_id=None):
    txt = find_txt(recharge_root, filename)
    if not txt:
        raise FileNotFoundError(f"❌ 未找到 {filename}.txt")

    print(f"📂 使用回灌文件: {txt}")
    data = load_jsonl(txt)

    frames = []
    values = []

    for fid in sorted(data.keys()):
        if start is not None and fid < start:
            continue
        if end is not None and fid > end:
            continue

        if obj_id is None:
            area, _ = calc_frame_total_area(data[fid])
        else:
            area = calc_frame_area_by_id(data[fid], obj_id)

        if area is None:
            continue

        frames.append(fid)
        values.append(area)

    print(f"📊 有效帧数: {len(frames)}")
    if not frames:
        print("⚠️ 无有效数据")
        return

    # ===== 画图 =====
    plt.figure(figsize=(12, 4))
    plt.plot(frames, values)
    plt.xlabel("Frame")
    plt.ylabel("rcBox Area (pixel²)")

    if obj_id is None:
        plt.title(f"Total rcBox Area | {filename}")
    else:
        plt.title(f"rcBox Area | ObjID={obj_id} | {filename}")

    plt.grid(True)
    plt.tight_layout()
    plt.show()


# =========================
# 入口
# =========================
if __name__ == "__main__":
    recharge_path = r"\\hz-iotfs02\Model_Test\TestSpace\Personal_Space\CHZ\Southlake\adas_perception_v3.1_SPC030_2m_98\output\shaoyuqi\V3.1_2M_3.1.27223.1398\20251226\lixiang1"

    while True:
        s = input(
            "输入:\n"
            "  filename\n"
            "  filename start end\n"
            "  filename start end obj_id\n"
            "(q 退出)\n"
        ).strip()

        if s.lower() == "q":
            break

        p = s.split()
        try:
            if len(p) == 1:
                plot_rcbox_area(recharge_path, p[0])
            elif len(p) == 3:
                plot_rcbox_area(recharge_path, p[0], int(p[1]), int(p[2]))
            elif len(p) == 4:
                plot_rcbox_area(
                    recharge_path,
                    p[0],
                    int(p[1]),
                    int(p[2]),
                    int(p[3])
                )
            else:
                print("❌ 参数错误")
        except Exception as e:
            print(f"❌ 出错: {e}")
