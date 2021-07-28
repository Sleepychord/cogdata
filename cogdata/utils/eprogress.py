#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on 2017/7/21

__author__ = 'HomgWu & Ming Ding'

import sys
import re
import abc
import threading

CLEAR_TO_END = "\033[K"
UP_ONE_LINE = "\033[F"


class ProgressBar(object, metaclass=abc.ABCMeta):
    """ Base module of all types of process bar.
    """

    def __init__(self, width=25, title=''):
        self.width = width
        self.title = ProgressBar.filter_str(title)
        self._lock = threading.Lock()

    @property
    def lock(self):
        return self._lock

    @abc.abstractmethod
    def update(self, progress=0):
        pass

    @staticmethod
    def filter_str(pending_str):
        """Filter \r、\t、\n"""
        return re.sub(pattern=r'\r|\t|\n', repl='', string=pending_str)


# class CircleProgress(ProgressBar):
#     def __init__(self, width=10, title=''):
#         """
#          @param width : 进度条展示的长度
#          @param title : 进度条前面展示的文字
#         """
#         super(CircleProgress, self).__init__(width=width, title=title)
#         self._current_char = ''

#     def update(self, progress=0):
#         """
#         @param progress : 当前进度值,非0则更新符号
#         """
#         with self.lock:
#             if progress > 0:
#                 self._current_char = self._get_next_circle_char(self._current_char)
#             sys.stdout.write('\r' + CLEAR_TO_END)
#             sys.stdout.write("\r%s:[%s]" % (self.title, self._current_char))
#             # sys.stdout.flush()

#     def _get_next_circle_char(self, current_char):
#         if current_char == '':
#             current_char = '-'
#         elif current_char == '-':
#             current_char = '\\'
#         elif current_char == '\\':
#             current_char = '|'
#         elif current_char == '|':
#             current_char = '/'
#         elif current_char == '/':
#             current_char = '-'
#         return current_char


class LineProgress(ProgressBar):
    """Normal Line progress bars.
    """

    def __init__(self, total=100, symbol='#', width=25, title=''):
        """
        Arguments
        ---------
        total:int
            The max number on progress bar
        symbol:str 
            The symbol of progress bar
        width:int 
            The number of symbols in the whole progress bar
        title:str
            Text before progress bar.
        """
        super(LineProgress, self).__init__(width=width, title=title)
        self.total = total
        self.symbol = symbol
        self._current_progress = 0

    def update(self, progress=0, speed=0):
        """
        Arguments
        ---------
        progress:int
            Current progress
        speed:float
            Average process speed(samples/s)
        """
        with self.lock:
            if progress > 0:
                self._current_progress = progress
            sys.stdout.write('\r' + CLEAR_TO_END)
            hashes = '#' * int(self._current_progress /
                               self.total * self.width)
            spaces = ' ' * (self.width - len(hashes))
            sys.stdout.write("\r%s:[%s] %d%%  Speed: %.2f samples/s" %
                             (self.title, hashes + spaces, self._current_progress, speed))
            # sys.stdout.flush()


class MultiProgressManager(object):
    def __new__(cls, *args, **kwargs):
        """Singleton"""
        if not hasattr(cls, '_instance'):
            cls._instance = super(MultiProgressManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        self._progress_dict = {}
        self._lock = threading.Lock()
        self.need_skip = False

    def put(self, key, progress_bar):
        """Add a progress bar
        Arguments
        ---------
        key:str
            The key of the new bar
        progress_bar:ProgressBar
            An instance of a progress bar
        """
        with self._lock:
            if key and progress_bar:
                self._progress_dict[key] = progress_bar
                progress_bar.index = len(self._progress_dict) - 1

    def clear(self):
        """Remove all progress bar"""
        with self._lock:
            self._progress_dict.clear()

    def update(self, key, progress, speed=0):
        """Update status of progress bars and repaint
        Arguments
        ---------
        key:str
            The key of the progress bar that need to update
        progress:int
            The current progress number
        speed:float
            Average process speed(samples/s)
        """
        with self._lock:
            if not key:
                return
            delta_line = len(self._progress_dict)
            if self.need_skip:
                self.need_skip = False
            else:
                sys.stdout.write(
                    UP_ONE_LINE * delta_line if delta_line > 0 else '')
            for tmp_key in self._progress_dict.keys():
                progress_bar = self._progress_dict.get(tmp_key)
                tmp_progress = 0
                if key == tmp_key:
                    tmp_progress = progress
                progress_bar.update(tmp_progress, speed=speed)
                sys.stdout.write('\n')

    def update_title(self, key, title):
        """Update the title of a progress bar
        Arguments
        ---------
        key:str
            The key of the progress bar that need to update title
        title:str
            New title
        """
        progress_bar = self._progress_dict.get(key, None)
        if progress_bar is not None:
            progress_bar.title = title

    def skip_upline(self):
        """Skip the first data line if True"""
        self.need_skip = True
