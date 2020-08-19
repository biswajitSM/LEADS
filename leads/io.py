from PyQt5.QtWidgets import QApplication, QFileDialog, QWidget


class FileDialog(QWidget):

    def __init__(self, title='File Dialog', filters="All Files (*)"):
        super().__init__()
        self.title = title
        self.filters = filters

    def openFileNameDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        filename, _ = QFileDialog.getOpenFileName(caption=self.title,
                                                  filter=self.filters, options=options)
        if filename:
            print(filename)
        return filename

    def openFileNamesDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        filenames, _ = QFileDialog.getOpenFileNames(self, self.title, directory=None,
                                                    filter=self.filters,
                                                    options=options)
        return filenames

    def saveFileDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        filename, _ = QFileDialog.getSaveFileName(caption=self.title,
                                                  filter=self.filters,
                                                  options=options)
        return filename

def get_fnames(directory='./', filters="Python Files (*.py)"):
    app = QApplication([])
    fd = FileDialog(filters=filters)
    filename = fd.openFileNamesDialog()
    return filename

if __name__ == "__main__":
    get_fnames()