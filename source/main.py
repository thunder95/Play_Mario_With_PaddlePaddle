__author__ = 'marble_xu'

from . import setup, tools
from . import constants as c
from .states import main_menu, load_screen, level

def main(que):
    game = tools.Control(que)
    state_dict = {c.MAIN_MENU: main_menu.Menu(),
                  c.LOAD_SCREEN: load_screen.LoadScreen(),
                  c.LEVEL: level.Level(),
                  c.GAME_OVER: load_screen.GameOver(),
                  c.TIME_OUT: load_screen.TimeOut()}
    game.setup_states(state_dict, c.MAIN_MENU)
    game.main()
