import numpy as np
from PIL import Image, ImageTk
import Tkinter as tk

class Radiograph(tk.Canvas):
    def __init__(self, parent, filename, hasLandmark=False):
        tk.Canvas.__init__(self, parent, bd=0, highlightthickness=0)
        self.parent = parent
        self.original = Image.open(filename)
        self.image = ImageTk.PhotoImage(self.original)
        self.create_image(0, 0, image=self.image, anchor=tk.NW, tags="IMG")

    def resize(self, event):
        size = (event.width, event.height)
        resized = self.original.resize(size, Image.ANTIALIAS)
        self.image = ImageTk.PhotoImage(resized)
        self.delete("IMG")
        self.create_image(0, 0, image=self.image, anchor=tk.NW, tags="IMG")
