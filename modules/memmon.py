import threading
import time
from collections import defaultdict

import torch


class MemUsageMonitor(threading.Thread):
    run_flag = None
    device = None
    disabled = False
    opts = None
    data = None

    def __init__(self, name, device, opts):
        threading.Thread.__init__(self)
        self.name = name
        self.device = device
        self.opts = opts

        self.daemon = True
        self.run_flag = threading.Event()
        self.data = defaultdict(int)

        try:
            self.cuda_mem_get_info()
            torch.cuda.memory_stats(self.device)
        except Exception as e:  # AMD or whatever
            print(f"Warning: caught exception '{e}', memory monitor disabled")
            self.disabled = True

    def cuda_mem_get_info(self):
        index = self.device.index if self.device.index is not None else torch.cuda.current_device()
        return torch.cuda.mem_get_info(index)

    def cuda_mem_get_info_safe(self):
        try:
            return self.cuda_mem_get_info()
        except Exception:
            return None

    def run(self):
        if self.disabled:
            return

        while True:
            self.run_flag.wait()

            torch.cuda.reset_peak_memory_stats()
            self.data.clear()

            if self.opts.memmon_poll_rate <= 0:
                self.run_flag.clear()
                continue

            mi0 = self.cuda_mem_get_info_safe()
            self.data["min_free"] = mi0[0] if mi0 is not None else 0

            while self.run_flag.is_set():
                mi = self.cuda_mem_get_info_safe()
                if mi is not None:
                    free, _total = mi
                    self.data["min_free"] = min(self.data["min_free"], free)

                time.sleep(1 / self.opts.memmon_poll_rate)

    def dump_debug(self):
        print(self, 'recorded data:')
        for k, v in self.read().items():
            print(k, -(v // -(1024 ** 2)))

        print(self, 'raw torch memory stats:')
        tm = torch.cuda.memory_stats(self.device)
        for k, v in tm.items():
            if 'bytes' not in k:
                continue
            print('\t' if 'peak' in k else '', k, -(v // -(1024 ** 2)))

        print(torch.cuda.memory_summary())

    def monitor(self):
        self.run_flag.set()

    def read(self):
        if not self.disabled:
            mi = self.cuda_mem_get_info_safe()
            if mi is not None:
                free, total = mi
                self.data["free"] = free
                self.data["total"] = total
                self.data["system_peak"] = total - self.data["min_free"]

            try:
                torch_stats = torch.cuda.memory_stats(self.device)
                self.data["active"] = torch_stats["active.all.current"]
                self.data["active_peak"] = torch_stats["active_bytes.all.peak"]
                self.data["reserved"] = torch_stats["reserved_bytes.all.current"]
                self.data["reserved_peak"] = torch_stats["reserved_bytes.all.peak"]
            except Exception:
                pass

        return self.data

    def stop(self):
        self.run_flag.clear()
        return self.read()
