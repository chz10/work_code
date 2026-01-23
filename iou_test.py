import os
import json
import matplotlib.pyplot as plt


# =====================================================
# rcBox 统一
# =====================================================
def normalize_rcbox(rc):
    if isinstance(rc, dict):
        return {
            "left": int(rc["left"]),
            "top": int(rc["top"]),
            "right": int(rc["right"]),
            "bottom": int(rc["bottom"]),
        }
    if isinstance(rc, list) and len(rc) == 4:
        return {
            "left": int(rc[0]),
            "top": int(rc[1]),
            "right": int(rc[2]),
            "bottom": int(rc[3]),
        }
    return None


def rcbox_area(rc):
    if not rc:
        return None
    w = rc["right"] - rc["left"]
    h = rc["bottom"] - rc["top"]
    return w * h if w > 0 and h > 0 else None


# =====================================================
# 工具：目录 → 自动找 obj 文件
# =====================================================
def resolve_obj_file(path):
    """
    允许：
    - 直接给文件
    - 给目录（自动找 *_obj*.txt / *.json）
    """
    if os.path.isfile(path):
        return path

    if not os.path.isdir(path):
        raise FileNotFoundError(f"路径不存在: {path}")

    for root, _, files in os.walk(path):
        for f in files:
            if (
                f.endswith(".txt")
                or f.endswith(".json")
            ):
                return os.path.join(root, f)

    raise FileNotFoundError(f"目录下未找到 obj 文件: {path}")


# =====================================================
# 核心解析
# =====================================================
def extract_objdata(path):
    path = resolve_obj_file(path)
    print(f"📂 使用文件: {path}")

    result = {}

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read().strip()

    # ---------- 整体 JSON ----------
    try:
        root = json.loads(raw)

        # ===== 实车 t JSON =====
        if isinstance(root, dict) and all(isinstance(v, list) for v in root.values()):
            for frame_str, objs in root.items():
                try:
                    frame = int(frame_str)
                except:
                    continue

                for obj in objs:
                    obj_id = obj.get("u8Id") or obj.get("s32ObjID")
                    rc = normalize_rcbox(obj.get("rcBox"))
                    if obj_id is None or not rc:
                        continue

                    result.setdefault(frame, {})
                    result[frame][int(obj_id)] = {
                        "obj_id": int(obj_id),
                        "rcBox": rc,
                    }

            print(f"📄 实车 t 解析完成 | frame={len(result)}")
            return result

        # ===== 回灌 JSON =====
        if "DecodeIneerOutput" in root:
            frame = int(root.get("frameId", 0))
            result.setdefault(frame, {})

            vpd = root["DecodeIneerOutput"].get("vpdOutput", {})

            def parse(lst, base):
                for i, o in enumerate(lst):
                    obj_id = o.get("s32ObjectId", base + i)
                    rc = normalize_rcbox(o.get("rcBox"))
                    if not rc:
                        continue
                    result[frame][int(obj_id)] = {
                        "obj_id": int(obj_id),
                        "rcBox": rc,
                    }

            parse(vpd.get("vehicleObjects", []), 10000)
            parse(vpd.get("vruObjects", []), 20000)

            print("📄 回灌 JSON 解析完成")
            return result

    except Exception:
        pass

    # ---------- 行 JSON / arcsoft ----------
    raw = raw.replace("}{", "}\n{")
    for line in raw.splitlines():
        try:
            j = json.loads(line)
        except:
            continue

        frame = j.get("frameId") or j.get("frameid")
        if frame is None:
            continue
        frame = int(frame)

        try:
            objs = j["Oem"]["visInfo"][0]["obj"]["objAttributes"]
        except:
            continue

        for o in objs:
            obj_id = o.get("u8Id") or o.get("s32ObjID")
            rc = normalize_rcbox(o.get("rcBox"))
            if obj_id is None or not rc:
                continue

            result.setdefault(frame, {})
            result[frame][int(obj_id)] = {
                "obj_id": int(obj_id),
                "rcBox": rc,
            }

    print(f"📄 arcsoft 解析完成 | frame={len(result)}")
    return result


# =====================================================
# 对比 & 绘图
# =====================================================
class RcBoxComparator:

    def compare(self, old_data, new_data, start, end, old_id, new_id):
        frames, a, b = [], [], []

        for f in range(start, end + 1):
            o = old_data.get(f, {}).get(old_id)
            n = new_data.get(f, {}).get(new_id)

            ao = rcbox_area(o["rcBox"]) if o else None
            an = rcbox_area(n["rcBox"]) if n else None

            if ao is None and an is None:
                continue

            frames.append(f)
            a.append(ao)
            b.append(an)

        return frames, a, b

    def plot(self, frames, a, b, old_id, new_id):
        plt.figure(figsize=(14, 4))
        plt.plot(frames, a, label=f"实车 ObjID={old_id}")
        plt.plot(frames, b, label=f"回灌 ObjID={new_id}")
        plt.xlabel("Frame")
        plt.ylabel("rcBox Area")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()


# =====================================================
# 入口
# =====================================================
if __name__ == "__main__":

    old_path = r"\\hz-iotfs02\Model_Test\TestSpace\Personal_Space\ZYF\SouthLake\adas_perception_v3.1_SPC030_2m\input\majianzhou\20260113"
    new_path = r"\\hz-iotfs02\Model_Test\TestSpace\Personal_Space\CHZ\Southlake\adas_perception_v3.1_SPC030_2m_98\output\shaoyuqi\V3.1_2M_3.1.27223.1398\20251226\lixiang1"

    old_data = extract_objdata(old_path)
    new_data = extract_objdata(new_path)

    tool = RcBoxComparator()
    frames, a, b = tool.compare(old_data, new_data, 0, 2000, 70, 105)
    tool.plot(frames, a, b, 70, 105)
