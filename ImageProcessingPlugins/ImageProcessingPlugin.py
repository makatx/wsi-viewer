'''
The below class should be extended in order to add 'Image Processing' add-on/plugin to the application
It provides access to the current file's name/location and viewport info. Actions can be run on the main application's tileManager.


Furthermore, the class also has a signal it can emit to share progress and display the same via the slot in the main 
application class
'''

from PyQt5.QtCore import QObject, pyqtSignal

class ImageProcessingPluginParent(QObject):
    def __init__(self, tileManager):
        super().__init__()
        ## tileManager property must be set, allow this class' __init__ to do that. tileManager object provides access to much of the slide info being viewed and also run actions on it
        self.tileManager = tileManager

        ## Do not override the below signal as this will be used my the main application to connect to its progress bar, updated when emitted
        ## the first 'str' argument is the string to display indicating progress info and the second argument, 'int' should share the % completion to reflect the 
        ## same in the process (runAction)
        ## emit this signal whenever progress bar needs to be updated. Last emit should send a 100 as the int argument.
        self.progressSignal = pyqtSignal(str, int, name='progressSignal')

        ## the below properties are set by the plugin object's __init__, overriding these placeholders below
        self.action_name = ""
        self.tooltip = ""

    def runAction(self):
        ## Define this function to perform the processing
        raise NotImplementedError('Plugin subclass must override the runAction() function')

