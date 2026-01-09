import subprocess
import sys
import os


class CheckerRunner:
    def __init__(self, module_or_script: str):
        """
        :param module_or_script: 模块文件路径（.py）或脚本文件路径
        """
        self.module_or_script = module_or_script

    def run_as_script(self, pseudo_true_value_path, recharge_res_path, checker_save_path, log_dir):
        """
        用 Popen 方式运行 checker 脚本，并把 stdout/stderr 重定向到 log 文件
        """
        os.makedirs(log_dir, exist_ok=True)
        pseudo_true_name = os.path.basename(pseudo_true_value_path)
        log_file_path = os.path.join(log_dir, f"{pseudo_true_name}.log")

        with open(log_file_path, "w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                [
                    sys.executable,
                    self.module_or_script,
                    pseudo_true_value_path,
                    recharge_res_path,
                    checker_save_path,
                ],
                stdout=log_file,
                stderr=subprocess.STDOUT,  # stderr 合并到 stdout
                text=True
            )

            process.wait()  # 等待脚本执行完成

        print(f"[Runner] Checker 执行完成，日志已保存: {log_file_path}")
        return process.returncode
