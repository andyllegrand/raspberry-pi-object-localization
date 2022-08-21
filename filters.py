
class sizeFilter:
    include_text = "SIZE"

    def __init__(self, args):
        object_size = args[0]
        self.min = 0
        self.max = 0

    def apply(self, contours):


