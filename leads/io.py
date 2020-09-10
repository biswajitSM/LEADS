from PyQt5.QtWidgets import QApplication, QFileDialog, QWidget
import os
import collections
import yaml

class FileDialog(QWidget):

    def __init__(self, directory=None, title='File Dialog', filters="All Files (*)"):
        super().__init__()
        self.directory = directory
        self.title = title
        self.filters = filters

    def openFileNameDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        filename, _ = QFileDialog.getOpenFileName(caption=self.title, directory=self.directory,
                                                  filter=self.filters, options=options)
        if filename:
            print(filename)
        return filename

    def openFileNamesDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        filenames, _ = QFileDialog.getOpenFileNames(self, self.title, directory=self.directory,
                                                    filter=self.filters,
                                                    options=options)
        return filenames

    def saveFileDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        filename, _ = QFileDialog.getSaveFileName(caption=self.title,
                                                  directory=self.directory,
                                                  filter=self.filters,
                                                  options=options)
        return filename

    def openDirectoryDialog(self):
        dirname = QFileDialog.getExistingDirectory(directory=self.directory)
        return dirname


def user_settings_filename():
    home = os.path.expanduser("~")
    return os.path.join(home, ".leads", "settings.yaml")

def load_user_settings():
    settings_filename = user_settings_filename()
    settings = None
    try:
        settings_file = open(settings_filename, "r")
    except FileNotFoundError:
        return AutoDict()
    try:
        settings = yaml.load(settings_file, Loader=yaml.FullLoader)
        settings_file.close()
    except Exception as e:
        print(e)
        print("Error reading user settings, Reset.")
    if not settings:
        return AutoDict()
    return AutoDict(settings)

def save_user_settings(settings):
    settings = to_dict_walk(settings)
    settings_filename = user_settings_filename()
    os.makedirs(os.path.dirname(settings_filename), exist_ok=True)
    with open(settings_filename, "w") as settings_file:
        yaml.dump(dict(settings), settings_file, default_flow_style=False)

def to_dict_walk(node):
    """ Converts mapping objects (subclassed from dict)
    to actual dict objects, including nested ones
    """
    node = dict(node)
    for key, val in node.items():
        if isinstance(val, dict):
            node[key] = to_dict_walk(val)
    return node

class AutoDict(collections.defaultdict):
    """
    A defaultdict whose auto-generated values are defaultdicts itself.
    This allows for auto-generating nested values, e.g.
    a = AutoDict()
    a['foo']['bar']['carrot'] = 42
    """

    def __init__(self, *args, **kwargs):
        super().__init__(AutoDict, *args, **kwargs)

if __name__ == "__main__":
    app = QApplication([])
    FileDialog().openFileNameDialog()