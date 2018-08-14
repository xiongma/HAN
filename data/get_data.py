
import os
import re
import xlrd
import xlwt
import pandas as pd

file_path = os.path.dirname(os.path.realpath(__file__))

class DataProcess(object):
    def read_data(self, file_path):
        """
        this function is able to read news from excel
        :return: data list
        """
        excel_file = xlrd.open_workbook(file_path)
        sheet = excel_file.sheet_by_index(0)
        data_list = []
        for i in range(0, sheet.nrows):
            sub_list = []
            title = sheet.cell_value(i, 0)
            content = sheet.cell_value(i, 1)
            label = sheet.cell_value(i, 2)
            if title == '' or len(content) < 100:
                continue
            title = self.regular_content(title)
            content = self.regular_content(content)
            sub_list.append(title)
            sub_list.append(content)
            sub_list.append(label)
            data_list.append(sub_list)

        return data_list

    def read_data_1(self, file_path):
        """
        this function is able to read news from excel
        :param file_path: file path
        :return: pandas data frame
        """
        data_frame = pd.read_excel(file_path)
        data_frame.columns = ['id', 'title', 'content']

        return data_frame

    def regular_content(self, title):
        """
        this function is able to regular content
        :param title: news title
        :return: regular title
        """
        title = ''.join(re.findall(u'[\u4e00-\u9fff]+', title))
        title = re.sub('|', '', title)

        return title

if __name__ == '__main__':
    dp = DataProcess()
    normal_news = dp.read_excel(r'F:\education\prediction.xls')
    garbage_news = dp.read_excel(r'F:\education\prediction.xls', True)
    dp.write_excel(r'F:\education\garbage_news.xls', garbage_news)
    dp.write_excel(r'F:\education\normal_news.xls', normal_news)