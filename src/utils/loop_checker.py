class LoopChecker():
    def __init__(self, location=None, threshold=1e6):
        self.location = location
        self.threshold = int(threshold)
        self.counter = 0
    
    def __call__(self):
        self.counter = (self.counter + 1) % self.threshold
        if(self.counter == 0):
            print("WARNING: LoopChecker in \"{}\" reached threshold {}".format(self.location, self.threshold))
