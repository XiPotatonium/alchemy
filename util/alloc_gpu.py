import os
import random
import time
from typing import List, Set


def alloc_cuda(
    preferred_devices: List[List[int]],
    wait_time: float = 240.0,
):
    """_summary_

    Args:
        preferred_devices (List[List[int]]): _description_
        wait_time (float, optional): _description_. Defaults to 240.0.

    Yields:
        _type_: _description_
    """
    import pynvml
    pynvml.nvmlInit()
    driver = pynvml.nvmlSystemGetCudaDriverVersion_v2()

    all_gpus = []
    if "ALL_GPU" in os.environ:
        all_gpus = list(map(int, os.environ["ALL_GPU"].split(",")))
    if len(all_gpus) == 0:
        # all_gpu not set by os.environ
        all_gpus = list(range(pynvml.nvmlDeviceGetCount()))

    gpu_queue: List[int] = []
    gpu_just_used: List[int] = []

    for task_preferred_devices in preferred_devices:
        task_preferred_devices: Set[int] = set(task_preferred_devices)
        if len(task_preferred_devices) == 1 and -1 in task_preferred_devices:
            # cpu-only
            yield {"device": "cpu"}
        else:
            # pick one gpu for training
            while len(gpu_queue) == 0:
                # detect free gpu if gpu queue is empty
                available_gpus = set(all_gpus) - set(gpu_just_used)
                if len(task_preferred_devices) == 0:
                    candidate_gpus = list(available_gpus)
                else:
                    # 只选择特定的GPU
                    candidate_gpus = list(task_preferred_devices.intersection(available_gpus))
                for index in candidate_gpus:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    mem_used = meminfo.used / 1024 / 1024   # MB
                    if mem_used < 1500:
                        # add free gpu
                        gpu_queue.append(index)
                if len(gpu_queue) == 0:
                    print("Preferred: {}. Waiting for Free GPU ......".format(
                        task_preferred_devices if len(task_preferred_devices) != 0 else available_gpus,
                    ))
                    time.sleep(wait_time)
                    gpu_just_used = []
                else:
                    print("Available device: ", gpu_queue)

            # print("########### Using GPU Normal Training ###########")

            device_id = random.sample(gpu_queue, k=1)[0]
            gpu_just_used.append(device_id)
            gpu_queue.remove(device_id)
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            yield {
                "device": "cuda:{}".format(device_id),
                "desc": str(pynvml.nvmlDeviceGetName(handle)),
                "driver": driver
            }
