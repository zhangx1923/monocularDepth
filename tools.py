import os
import configparser
import matplotlib.pyplot as plt
from visdom import Visdom

#记录执行的日志，包括正常执行的输出以及发生错误的error信息
class Log:
    #file_location应该是文件夹的路径，而不是文件路径，文件夹路径最后不加“/”
    def __init__(self, file_location):
        if not os.path.exists(file_location):
            os.mkdir(file_location)
        self.log_loc = file_location + "/log.txt"
        self.err_loc = file_location + "/error.txt"
    
    #msg: 写入文件的内容,写入log file
    def print(self, msg):
        with open(self.log_loc, "w", buffering=1) as f:
            f.write(msg + "\r\n")
            f.flush()
            #终端同步提醒
            print(msg)

    #msg: 写入文件的内容，写入error file
    def error(self, msg):
        with open(self.err_loc, "w", buffering=1) as f:
            f.write(msg + "\r\n")
            f.flush()
            #终端同步提醒
            print(msg)
        exit(0)

#与配置文件交互
class Conf:
    def __init__(self, path = "configure.ini", section = "monocular") -> None:
        self.conf = configparser.ConfigParser()
        self.conf.read(path)
        self.section = section
    
    def getContent(self, key_str):
        return self.conf.get(self.section, key_str)


#可视化功能
class Visualization:
    def __init__(self, env = "monocular") -> None:
        self.viz = Visdom(env=env)
        assert self.viz.check_connection(), \
            "No connection could be found!"
        
        #visdom中显示文字的窗口
        self.txt_win = None
        #显示散点图的窗口
        self.sca_win = None
        #显示曲线的窗口
        self.line_win = None
        #显示柱状体的窗口
        self.bar_win = None

        self.env_name = env

    #曲线
    def line(self, X, y):
        if self.line_win == None:
            self.line_win = self.viz.line(X,y)
        else:
            self.viz.scatter(X,y,win=self.line_win, update="new")

    #散点图
    def scatter(self, X, y):
        if self.sca_win == None:
            self.sca_win = self.viz.scatter(X,y)
        else:
            self.viz.scatter(X,y,win=self.sca_win, update="new")

    #显示文字
    def text(self, msg):
        if self.txt_win == None:
            self.txt_win = self.viz.text(msg)
        else:
            self.viz.text(msg, win=self.txt_win, append=True)

    #柱形图
    def bar(self, X, y):
        if self.bar_win == None:
            self.bar_win = self.viz.bar(X,y)
        else:
            self.viz.bar(X,y, win=self.bar_win, update="new")

    def save(self):
        self.viz.save([self.env_name])
