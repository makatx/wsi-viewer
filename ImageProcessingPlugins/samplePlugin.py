from ImageProcessingPlugins.ImageProcessingPlugin import ImageProcessingPluginParent

class samplePlugin(ImageProcessingPluginParent):
    def __init__(self, tileManager):
        super().__init__(tileManager)
        self.action_name = "sample action"
        self.tooltip = "This is just a sample plugin action"

    def runAction(self):
        print("this being printed from sample plugin")