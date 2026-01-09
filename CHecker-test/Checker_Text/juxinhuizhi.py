import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 输入数据
data = {
    "time_stamp": 1756274902276,
    "frame_id": 1394,
    "GT_type": 0,
    "generated_objects": [
        {
            "ID": 77,
            "obstacles_type": 0,
            "longDistance": 10.503912925720215,
            "latDistance": -0.6809433102607727,
            "length": 7.0,
            "width": 2.40828275680542,
            "height": 3.1200551986694336,
            "heading": -4.0774805916837997e-22,
            "absolute_longitudinal_velocity": 2.5042184771643414e-19,
            "absolute_longitudinal_acceleration": 0.0,
            "absolute_lateral_velocity": 0.0,
            "absolute_lateral_acceleration": 0.0
        }
    ]
}

# 取第一个对象
obj = data["generated_objects"][0]

# 获取矩形中心点与尺寸
cx = obj["longDistance"]
cy = obj["latDistance"]
length = obj["length"]
width = obj["width"]

# heading 基本为 0，这里先不旋转
heading = obj["heading"]

# 由中心点转换为左下角点
x = cx - length / 2
y = cy - width / 2

# 创建图形
fig, ax = plt.subplots()

# 矩形（无旋转）
rect = patches.Rectangle(
    (x, y),
    length,
    width,
    fill=False,
    edgecolor='black'
)
ax.add_patch(rect)

# 展示范围（自动扩展）
ax.set_xlim(cx - 10, cx + 10)
ax.set_ylim(cy - 10, cy + 10)
ax.set_aspect('equal', 'box')

plt.xlabel("Longitudinal (m)")
plt.ylabel("Lateral (m)")
plt.title("Object Rectangle")

plt.grid(True)
plt.show()
