import Tkinter as tk

from radiograph import Radiograph

class Application(tk.Tk):
    def __init__(self, parent=None):
        tk.Tk.__init__(self, parent)
        self.parent = parent
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.bind("<Configure>", self.resize)
        self.bind("<Control-c>", lambda e: self.quit())

        self.radiographs = [Radiograph(self, "data/radiographs/01.tif")]
        self.showId = 0
        self.radiographs[self.showId].grid(row=0, sticky=tk.W+tk.E+tk.N+tk.S)

    def resize(self, event):
        self.radiographs[self.showId].resize(event)
