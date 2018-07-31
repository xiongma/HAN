
import os

file_path = os.path.dirname(os.path.realpath(__file__))

class Political(object):
    def __init__(self):
        # political news
        self.all_path = file_path + '\\ds\\political\\political_data.xls'
        self.normal_path = file_path + '\\ds\\political\\political_normal_data.xls'
        self.garbage_path = file_path + '\\ds\\political\\political_garbage_data.xls'
        self.corruption_path = file_path + '\\ds\\political\\political_corruption_data.xls'

class Advertising(object):
    def __init__(self):
        # advertising news
        self.all_path = file_path + '\\ds\\advertising\\advertising_data.xls'
        self.normal_path = file_path + '\\ds\\advertising\\advertising_normal_data.xls'
        self.garbage_path = file_path + '\\ds\\advertising\\advertising_garbage_data.xls'

class Education(object):
    def __init__(self):
        # education news
        self.all_path = file_path + '\\ds\\education\\education_data.xls'
        self.normal_path = file_path + '\\ds\\education\\education_normal_data.xls'
        self.garbage_path = file_path + '/ds/education/education_garbage_data.xls'