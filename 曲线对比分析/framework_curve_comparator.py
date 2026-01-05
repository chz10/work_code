from GetSpecificAttributes import *
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置全局中文字体（Windows 常用 SimHei，Linux 常用 DejaVu Sans/思源黑体）
rcParams['font.sans-serif'] = ['SimHei']   # 黑体
rcParams['axes.unicode_minus'] = False     # 解决坐标轴负号显示问题

class FrameworkCurveComparator:
    def __init__(self, video_path, recharge_path, version_old, version_new):
        self.video_path = video_path
        self.recharge_path = recharge_path
        self.version_old = version_old
        self.version_new = version_new

        # 缓存搜索结果
        self._video_filelist = None
        self._recharge_filelist = None

    def find_txt_files(self, folder_path, extension=".txt"):
        """
        查找指定文件夹下的所有.h264文件。
        :param folder_path: 文件夹路径
        :return: .h264文件路径列表
        """
        txt_filelist = {}
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith(extension):
                    filename = os.path.splitext(file)[0]
                    filedirname = os.path.splitext(os.path.basename(os.path.dirname(root)))[0]
                    if filename not in txt_filelist:
                        txt_filelist[filename] = os.path.join(root, file)
                        
        return txt_filelist
    
    # 评价抖动程度（暂未使用）
    def classify_jitter_diff(self, data, fs=1.0, amp_thresh=0.05, freq_thresh=0.1):
        data = np.array(data, dtype=float)
        diff = np.diff(data)
        
        # 主导频率
        freqs = np.fft.rfftfreq(len(diff), d=1/fs) # 各个频率分量的强度
        spectrum = np.abs(np.fft.rfft(diff)) # 差分信号的频谱
        f_dom = freqs[np.argmax(spectrum[1:])+1] # 主导频率
        freq_label = "低频" if f_dom < freq_thresh else "高频"

        # 抖动幅度
        rms_diff = np.sqrt(np.mean(diff**2))
        amp_label = "低幅" if rms_diff < amp_thresh else "高幅"
        
        return f"{self.find_txt_files}{freq_label}{amp_label}抖动", f_dom, rms_diff

    def load_filelists(self):
        """只搜索一次并缓存"""
        if self._video_filelist is None:
            self._video_filelist = self.find_txt_files(self.video_path, extension=".txt")
        if self._recharge_filelist is None:
            self._recharge_filelist = self.find_txt_files(self.recharge_path, extension=".txt")
    
    def run_analysis(self, filename, start_frame, end_frame, target_id_old, target_id_new):
        self.load_filelists() # 确保文件列表已加载
        self.resultdata_old = []
        self.resultdata_new = []
        self.filename = filename
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.target_id_old = target_id_old
        self.target_id_new = target_id_new

        videoapth = self._video_filelist.get(self.filename)
        # obj_txt = os.path.join(os.path.dirname(videoapth), "log", self.filename + "_arcsoft_obj.txt")
        recharge_txt = self._recharge_filelist.get(self.filename)
        arc_objdata_old = extract_objdata_arc(videoapth)
        arc_objdata_new = extract_objdata_arc(recharge_txt)

        self.compare_curves(arc_objdata_old, arc_objdata_new)
    
    def compare_curves(self, arc_objdata_old, arc_objdata_new):
        frame_list = list(range(self.start_frame, self.end_frame))

        # plt_list = ["LongDistance", "LatDistance", 
        #             "AbsoluteLongVelocity", "AbsoluteLatVelocity", 
        #             "RelativeLatVelocity", "RelativeLongVelocity", 
        #             "Box_width", "Box_height"]

        plt_list = ["LongDistance", "LatDistance", 
                    "AbsoluteLatVelocity", "AbsoluteLongVelocity"]

        # 创建 2x4 子图
        fig, axes = plt.subplots(2, 2, figsize=(18, 9))
        axes = axes.flatten()  # 转成一维，方便索引

        for idx, key in enumerate(plt_list):
            value_list_old = []
            value_list_new = []

            testlist = []

            for i in frame_list:
                try:
                    if key == "Box_width":
                        target_rcBoxdict = arc_objdata_old[i][self.target_id_old]['rcBox']
                        left = target_rcBoxdict.get("left")
                        right = target_rcBoxdict.get("right")
                        value = right - left
                        value_list_old.append(value if value is not None else np.nan)
                        testlist.append(value)

                    elif key == "Box_height":
                        target_rcBoxdict = arc_objdata_old[i][self.target_id_old]['rcBox']
                        bottom = target_rcBoxdict.get("bottom")
                        top = target_rcBoxdict.get("top")
                        value = bottom - top
                        value_list_old.append(value if value is not None else np.nan)
                        testlist.append(value)

                    else:
                        frame_data = arc_objdata_old[i]
                        value = arc_objdata_old[i][self.target_id_old][key]
                        value_list_old.append(value if value is not None else np.nan)
                        testlist.append(value)
                except Exception:
                    value_list_old.append(np.nan)

            for i in frame_list:
                if key == "Box_width":
                    target_rcBoxdict = arc_objdata_new[i][self.target_id_new]['rcBox']
                    left = target_rcBoxdict.get("left")
                    right = target_rcBoxdict.get("right")
                    value = right - left
                    value_list_new.append(value if value is not None else np.nan)
                    testlist.append(value)

                elif key == "Box_height":
                    target_rcBoxdict = arc_objdata_new[i][self.target_id_new]['rcBox']
                    bottom = target_rcBoxdict.get("bottom")
                    top = target_rcBoxdict.get("top")
                    value = bottom - top
                    value_list_new.append(value if value is not None else np.nan)
                    testlist.append(value)

                else:
                    if i in arc_objdata_new:
                        frame_data = arc_objdata_new[i]
                    else:
                        continue
                    if self.target_id_new in frame_data:
                        value = arc_objdata_new[i][self.target_id_new][key]
                    else:
                        value = None
                    value_list_new.append(value if value is not None else np.nan)
                    testlist.append(value)
            self.plot_framework_curves(frame_list, value_list_old, value_list_new, key, axes[idx])

        plt.tight_layout()
        plt.show()
        # plt.close()

    def plot_framework_curves(self, frame_list, data_old, data_new, ylabel, ax):
        data_old = np.array(data_old, dtype=float)
        data_new = np.array(data_new, dtype=float)
        frame_list = np.array(frame_list, dtype=int)

        # 处理数据不连续：根据原始帧号插入 NaN
        def insert_nan_if_discontinuous(frame_nums, values):
            x_new, y_new = [], []
            last_frame = None
            for f, v in zip(frame_nums, values):
                if v is None or np.isnan(v):
                    x_new.append(f)
                    y_new.append(np.nan)
                else:
                    if last_frame is not None and f != last_frame + 1:
                        x_new.append(f - 0.5)
                        y_new.append(np.nan)
                    x_new.append(f)
                    y_new.append(v)
                    last_frame = f
            return np.array(x_new), np.array(y_new)

        x_2_2, y_2_2 = insert_nan_if_discontinuous(frame_list, data_old)
        x_2_3, y_2_3 = insert_nan_if_discontinuous(frame_list, data_new)

        # # 计算一阶差分标准差（剔除 NaN）
        def compute_diff_std_safe(arr):
            arr = arr[~np.isnan(arr)]
            if len(arr) < 2:
                return np.nan
            return np.std(np.diff(arr))

        std_2_2 = compute_diff_std_safe(y_2_2)
        std_2_3 = compute_diff_std_safe(y_2_3)

        # 把 STD 放进 label
        ax.plot(x_2_2, y_2_2, label=f"{self.version_old} (STD={std_2_2:.6f})", color='b')
        ax.plot(x_2_3, y_2_3, label=f"{self.version_new} (STD={std_2_3:.6f})", color='g')

        ax.set_xlabel("Frame ID")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel}")
        ax.legend()
        ax.grid(True)


if __name__ == "__main__":
    video_path = r"\\hz-iotfs02\Model_Test\TestSpace\Personal_Space\CHZ\Southlake\adas_perception_v3.1_SPC030_2m\output\gaoziyi\20251208\lxiang2"
    recharge_path = r"\\hz-iotfs02\Model_Test\TestSpace\Personal_Space\CHZ\Southlake\adas_perception_v3.1_SPC030_2m_42\output_test\lishun\V3.1_2M_3.1.27223.1349_VPT_4701\20251212\test"
    version_old = "V3.1.27223.1272"
    version_new = "V3.1.27223.1348"
    analyzer = FrameworkCurveComparator(video_path, recharge_path, version_old, version_new)

    while True:
        # try:
        print("请输入参数(输入 q 退出) (格式: filename start_frame end_frame target_id_old target_id_new)")
        user_input = input("输入示例: VisInsight_20250821093802 2100 2300 136 45 \n").strip()
        if user_input.lower() == "q":
            break

        parts = user_input.split()
        if len(parts) != 5:
            print("❌ 参数数量不正确，请重新输入!")
            continue

        filename, start_frame, end_frame, target_id_old, target_id_new = parts
        start_frame = int(start_frame)
        end_frame = int(end_frame)
        target_id_old = int(target_id_old)
        target_id_new = int(target_id_new)

        analyzer.run_analysis(filename, start_frame, end_frame, target_id_old, target_id_new)
        print(f"✅ {filename} 分析完成!")

        # except Exception as e:
        #     print(f"❌ 出错: {e}")