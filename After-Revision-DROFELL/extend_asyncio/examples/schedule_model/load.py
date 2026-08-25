# import threading
# import time
import extend_asyncio as asyncio


def work(load: float, name: str):
    asyncio.work_load_simulate(load, name)

# active_lock = threading.Lock()
# active_works = set()  # 存储当前活跃线程的 ID
# SIMULATE_STEP = 0.001/10
# PENALTY_RATIO = 1.0


# def work(duration: float):
#     tid = threading.get_ident()
#     completed = 0.0

#     with active_lock:
#         active_works.add(tid)

#     last_time = time.time()
#     while completed < duration:
#         new_time = time.time()
#         while new_time-last_time < SIMULATE_STEP:
#             new_time = time.time()

#         real_step = new_time - last_time
#         last_time = new_time

#         with active_lock:
#             others = len(active_works) - 1

#         # 当前线程的执行速率：越多线程越慢
#         exec_rate = 1.0 / (1.0 + PENALTY_RATIO * others)
#         completed += real_step * exec_rate

#         print(f"[{tid}] {completed:.4f}/{duration:.4f} (others={others})", end="\r")
#     with active_lock:
#         active_works.discard(tid)


# if __name__ == "__main__":
#     def worker(id: int, t: float):
#         st = time.time()
#         complete_times = 0
#         while time.time()-st < 5:
#             asyncio.work_load_simulate(t, f"{id}")
#             # work(t)
#             complete_times += 1
#         print(f"task: {id} finish {complete_times}, each run: {(time.time()-st)/complete_times:.6f}s, slow down: {(time.time()-st)/(complete_times*t):.2%}")

#     threads = [
#         threading.Thread(target=worker, args=(1, 0.002)),
#         threading.Thread(target=worker, args=(2, 0.003)),
#     ]

#     for t in threads:
#         t.start()

#     for t in threads:
#         t.join()
