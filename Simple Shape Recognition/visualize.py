from tkinter import *
from colorsys import hsv_to_rgb
import numpy as np
from math import e

tk=Tk()
scale_factor=8

canvas=Canvas(tk, width=scale_factor*65, height=scale_factor*65)
canvas.pack()
tk.update()


def draw_pixel(x, y, color):
    canvas.create_rectangle(scale_factor*x, scale_factor*y, scale_factor*(x+1), scale_factor*(y+1), fill=color, outline="")

def normalize(M):
    #return e**M/np.sum(e**M)
    return 100*(M-M.min())/(M.max()-M.min())

def visualize(M, size=65):
    for x in range(0, size):
        for y in range(0, size):
            pixel=M[size*x+y]
            draw_pixel(x, y, "#"+"".join(["{:02x}".format(z) for z in hsv_to_rgb(0, 0, round(pixel))]))

M=np.load("L1_to_L2.npy")
visualize(normalize(M))
