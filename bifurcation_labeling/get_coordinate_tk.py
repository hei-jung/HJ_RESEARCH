from tkinter import *
import numpy as np
from PIL import Image, ImageTk


def callback(event):
    canvas = event.widget
    x = canvas.canvasx(event.x)
    y = canvas.canvasy(event.y)
    print(canvas.find_closest(x, y))


root = Tk()
root.geometry("1000x800+100+100")
root.resizable(True, True)

canvas = Canvas(root, width=800, height=800)
canvas.bind("<Button-1>", callback)
canvas.pack()

img_path = '/home/jhj/Desktop/labeling/mip_coronal/14090110_20180523_095025_MR.npy'
np_img = np.load(img_path)
np_img = (np_img - np_img.min()) / (np_img.max() - np_img.min())
np_img = np_img * 255
img = ImageTk.PhotoImage(image=Image.fromarray(np_img.squeeze()))
canvas.create_image(20, 20, anchor=NW, image=img)

root.mainloop()
