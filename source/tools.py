__author__ = 'marble_xu'

import os
import pygame as pg
from abc import ABC, abstractmethod

keybinding = {
    'action': 0,
    'jump': 1,
    'left': 2,
    'right': 3,
    'up': 4,
    'down': 5,
    "enter": 6,
}

class State():
    def __init__(self):
        self.start_time = 0.0
        self.current_time = 0.0
        self.done = False
        self.next = None
        self.persist = {}
    
    @abstractmethod
    def startup(self, current_time, persist):
        '''abstract method'''

    def cleanup(self):
        self.done = False
        return self.persist
    
    @abstractmethod
    def update(sefl, surface, keys, current_time):
        '''abstract method'''

class Control():
    def __init__(self, q):
        self.screen = pg.display.get_surface()
        self.done = False
        self.clock = pg.time.Clock()
        self.fps = 30
        self.current_time = 0.0
        self.keys = [0, 0, 0, 0, 0, 0, 0]
        self.state_dict = {}
        self.state_name = None
        self.state = None
        self.action_queue = q
    
    def setup_states(self, state_dict, start_state):
        self.state_dict = state_dict
        self.state_name = start_state
        self.state = self.state_dict[self.state_name]
    
    def update(self):
        self.current_time = pg.time.get_ticks()
        if self.state.done:
            self.flip_state()
        self.state.update(self.screen, self.keys, self.current_time)
    
    def flip_state(self):
        previous, self.state_name = self.state_name, self.state.next
        persist = self.state.cleanup()
        self.state = self.state_dict[self.state_name]
        self.state.startup(self.current_time, persist)

    def convert_keyboard_keys(self, raw_keys):
        self.keys[keybinding["action"]] = 1 if raw_keys[pg.K_s] else 0
        self.keys[keybinding["jump"]] = 1 if raw_keys[pg.K_a] else 0
        self.keys[keybinding["left"]] = 1 if raw_keys[pg.K_LEFT] else 0
        self.keys[keybinding["right"]] = 1 if raw_keys[pg.K_RIGHT] else 0
        self.keys[keybinding["up"]] = 1 if raw_keys[pg.K_UP] else 0
        self.keys[keybinding["down"]] = 1 if raw_keys[pg.K_DOWN] else 0
        self.keys[keybinding["enter"]] = 1 if raw_keys[pg.K_RETURN] else 0

    def event_loop(self):
        is_keyboard = False
        raw_keys = None
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.done = True
            elif event.type == pg.KEYDOWN:
                raw_keys = pg.key.get_pressed()
            elif event.type == pg.KEYUP:
                raw_keys = pg.key.get_pressed()
            is_keyboard = True #键盘模式

        if is_keyboard and raw_keys is not None:
            self.convert_keyboard_keys(raw_keys)
        else:
            while not self.action_queue.empty():
                # res = self.action_queue.get_nowait()
                # self.keys = [int(x) for x in res.split(",")]
                self.keys = self.action_queue.get_nowait()


    def main(self):
        while not self.done:
            self.event_loop()
            self.update()
            pg.display.update()
            self.clock.tick(self.fps)

def get_image(sheet, x, y, width, height, colorkey, scale):
        image = pg.Surface([width, height])
        rect = image.get_rect()

        image.blit(sheet, (0, 0), (x, y, width, height))
        image.set_colorkey(colorkey)
        image = pg.transform.scale(image,
                                   (int(rect.width*scale),
                                    int(rect.height*scale)))
        return image

def load_all_gfx(directory, colorkey=(255,0,255), accept=('.png', '.jpg', '.bmp', '.gif')):
    graphics = {}
    for pic in os.listdir(directory):
        name, ext = os.path.splitext(pic)
        if ext.lower() in accept:
            img = pg.image.load(os.path.join(directory, pic))
            if img.get_alpha():
                img = img.convert_alpha()
            else:
                img = img.convert()
                img.set_colorkey(colorkey)
            graphics[name] = img
    return graphics
