import time
import json
import contextlib
import numpy as np
import types
from collections import UserDict


class Profiler():
    """Class for profiling execution time of code."""

    def __init__(self):
        # simple profiling
        self._start = None
        self._end = None
        # named profiling
        self._records = {}
        self.__names_context = [] # stack storing names of disjoint records for nested profiling contextmanagers

    def start(self, name=None):
        """Start timer of profiler."""
        if name is None:
            # simple profiling
            self._start = time.time()
            self._end = None
        else:
            # named profiling
            if name not in self._records:
                self._records[name] = {
                    "start": None, 
                    "end": None, 
                    "time": 0.0, 
                    "info": None
                }
            self._records[name]["start"] = time.time()
            self._records[name]["end"] = None

    def stop(self, name=None):
        """Stop timer of profiler."""
        if name is None:
            # simple profiling
            if self._start is None:
                print("WARNING: profiler has not been started yet")
                return
            self._end = time.time()
        else:
            # named profiling
            if name not in self._records:
                print("WARNING: profiler has not been started for \"{}\" yet".format(name))
                return
            self._records[name]["end"] = time.time()

    def time(self, name=None):
        """Return last stopped time."""
        if name is None:
            # simple profiling
            if self._start is None or self._end is None:
                print("WARNING: profiler has not recorded a time yet")
                return
            return self._end - self._start
        else:
            # named profiling
            if name not in self._records or self._records[name]["start"] is None or self._records[name]["end"] is None:
                print("WARNING: profiler has not recorded a time for \"{}\" yet".format(name))
                return
            return self._records[name]["end"] - self._records[name]["start"]

    def save(self, name):
        """Save last stopped time under given name cumulatively."""
        if name not in self._records:
            self._records[name] = {
                "start": self._start,
                "end": self._end,
                "time": self.time(),
                "info": None,
            }
        else:
            self._records[name]["time"] += self.time(name)
        
    def set_info(self, name, info):
        """Set profiling information for the stopped time under given name."""
        self._records[name]["info"] = info
        
    def merge(self, profiler):
        """Merge records with another profiler."""
        self._records.update(profiler._records)
    
    def reset(self):
        """Reset profiler and delete all records."""
        self._start = None
        self._end = None
        self._records.clear()
        
    def print(self, names=None):
        """Print saved times and profiling information for the given list of names."""
        if names is None:
            names = self._records.keys()
                
        print("Profiling")
        max_length = np.max([len(name) for name in names])
        total = 0
        for name in names:
            time = self._records[name]["time"] if name in self._records else None
            info = self._records[name]["info"] if name in self._records else None
            total += time if time is not None else 0
            print("  {:{}} {}{}".format(
                "{}:".format(name), max_length + 1,
                "{:4.2f}s".format(time) if time is not None else "{:4} ".format("-"),
                " ({})".format(info) if info is not None else "",
            ))
        print("  {:{}} {}".format("Total:", max_length + 1, "{:4.2f}s".format(total)))
        
    @contextlib.contextmanager
    def cm(self, name=None, disjoint=True):
        """Create context manager for profiling.
        
        Args:
            name (str): Name of the profiling session.
            disjoint (bool): Flag whether recorded time should be disjoint from
                other times recorded with context manager. Defaults to True."""
        if name is None:
            # simple profiling
            self.start()
            try:
                yield self
            finally:
                self.stop()
        else:
            # named profiling
            # stop previous session
            if len(self.__names_context) > 0:
                name_prev = self.__names_context[-1]
                self.stop(name=name_prev)
                self.save(name_prev)
            # start this session
            if disjoint:
                self.__names_context.append(name)
            self.start(name=name)
            try:
                yield self
            finally:
                # stop this session
                self.stop(name=name)
                self.save(name)
                if disjoint:
                    self.__names_context.pop()
                # start previous session
                if len(self.__names_context) > 0:
                    name_prev = self.__names_context[-1]
                    self.start(name=name_prev)


class LoopChecker():
    """Class for checking loops on non-termination."""
    def __init__(self, location=None, threshold=1e6):
        self.location = location
        self.threshold = int(threshold)
        self.counter = 0
    
    def __call__(self):
        self.counter = (self.counter + 1) % self.threshold
        if(self.counter == 0):
            print("WARNING: LoopChecker in \"{}\" reached threshold {}".format(self.location, self.threshold))


class LazyDict(UserDict):
    """Class for lazily evaluating function items of dict on first access."""
    def __getitem__(self, key):
        if isinstance(self.data[key], types.FunctionType):
            self.data[key] = self.data[key]()
        return self.data[key]


def build_json_encoder(encoders=[]):
    """Encode common non-serializable types into serializable ones."""
    def encoder(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        for instance, encoder in encoders:
            if isinstance(obj, instance):
                return encoder(obj)
        return json.JSONEncoder().default(obj)
    return encoder
