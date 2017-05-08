import Tkinter as tk
from PIL import Image, ImageTk

class Application(tk.Tk):
    def __init__(self, parent=None):
        tk.Tk.__init__(self, parent)
        self.parent = parent
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.original = Image.open("data/radiographs/01.tif")
        self.image = ImageTk.PhotoImage(self.original)
        self.display = tk.Canvas(self, bd=0, highlightthickness=0)
        self.display.create_image(0, 0, image=self.image, anchor=tk.NW, tags="IMG")
        self.display.grid(row=0, sticky=tk.W+tk.E+tk.N+tk.S)
        self.bind("<Configure>", self.resize)
        self.bind("<Control-c>", lambda e: self.quit())

    def resize(self, event):
        size = (event.width, event.height)
        resized = self.original.resize(size, Image.ANTIALIAS)
        self.image = ImageTk.PhotoImage(resized)
        self.display.delete("IMG")
        self.display.create_image(0, 0, image=self.image, anchor=tk.NW, tags="IMG")
