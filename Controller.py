from FilterSegmentation import FilterSegmentation

class Controller:
    def run_filter(self, image):
        return FilterSegmentation.run(image)
    
    def run_ml(self, image):
        pass

    def run_dl(self, image):
        pass