from datetime import datetime
import time

while True:
    now = datetime.now()
    current_time = now.strftime("%Y%m%d_%H%M")
    print("\r当前时间是：", current_time, end="")
    time.sleep(1)  # 每隔1秒更新一次