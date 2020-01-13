from ImageProcessingPlugins.ImageProcessingPlugin import ImageProcessingPluginParent
from time import sleep

class samplePlugin(ImageProcessingPluginParent):
    def __init__(self, tileManager):
        super().__init__(tileManager)
        self.action_name = "sample action"
        self.tooltip = "This is just a sample plugin action"

    def runAction(self):
        print("this being printed from sample plugin")
        self.progressSignal.emit("Emitting Once...", 30)
        for i in range(6):
            sleep(1)
            self.progressSignal.emit("Emitting again...", 30+i*10)

