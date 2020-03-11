# almost always will import these
import sys
from PyQt4 import QtGui

# need application definition; allows passing arguments when calling from the command line
app = QtGui.QtApplication(sys.argv)

# define the window
window = QtGui.QWidget()

window.show()
