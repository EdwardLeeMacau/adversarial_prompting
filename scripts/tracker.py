# Stub code to replace WandB with a local tracker

import os
import json
import numpy as np
import time
from typing import Dict

class Run:
    def __init__(self):
        self.name = "local"
        self.start_time = time.time()

class Tracker:
    def __init__(self, config: Dict, **kwargs):
        self.config = config
        # self.run = Run()

    def __repr__(self) -> str:
        print(json.dumps(self.config, indent=2))

    @staticmethod
    def init(**kwargs):
        return Tracker(**kwargs)

    def log(self, data: Dict, step: int = None):
        # Know problems:
        # Object of type Tensor is not JSON serializable.
        for k, v in data.items():
            if isinstance(v, list):
                data[k] = np.array(v)

        print(data)

    def finish(self):
        pass