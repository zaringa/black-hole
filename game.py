import pygame as py
py.init()
py.display.set_mode((800,600))

run = True
while run:
    for i in py.event.get():
        if i.type == py.QUIT:
            run = False

py.quit()