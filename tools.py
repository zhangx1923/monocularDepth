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
    def __init__(self) -> None:
        pass
