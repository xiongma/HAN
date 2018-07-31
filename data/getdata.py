import os
import re
import xlrd
import xlwt
import dictjieba

file_path = os.path.dirname(os.path.realpath(__file__))

# class AreaJieba(object):
#     def __init__(self):
#         """
#         init area jieba
#         """
#         import areajieba
#         areajieba.initialize(dictjieba.area_dict)
#         areajieba.load_userdict(dictjieba.area_user_dict)
# 
#         from areajieba.analyse import extract_tags as area_extract_tags
#         self.area_extract_tags = area_extract_tags

class TagJieba(object):
    def __init__(self):
        """
        init tag jieba
        """
        import tagjieba
        tagjieba.initialize(dictjieba.slda_tag_dict)

        from tagjieba.analyse import extract_tags as tag_extract_tags
        from tagjieba.analyse import set_stop_words as tag_set_stop_words
        tag_set_stop_words(dictjieba.stop_dict)
        self.tag_extract_tags = tag_extract_tags
        self.lcut = tagjieba.lcut

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
            if title == '' or len(content) == 0:
                continue
            title = self.regular_content(title)
            content = self.regular_content(content)
            sub_list.append(title)
            sub_list.append(content)
            data_list.append(sub_list)

        return data_list

    def read_excel(self, file_path, label):
        """
        this function is able to read  regional political and normal news from all data excel
        :param file_path: all data excel
        :param label: data label, it's float
        :return: data list
        """
        excel_file = xlrd.open_workbook(file_path)
        sheet = excel_file.sheet_by_index(0)
        data = []
        for i in range(1, sheet.nrows):
            sub_list = []
            flag = sheet.cell_value(i, 3)
            if flag == label:
                title = sheet.cell_value(i, 1)
                title = self.regular_content(title)
                content = sheet.cell_value(i, 2)
                if len(content) > 10:
                    sub_list.append(title)
                    sub_list.append(content)
                    data.append(sub_list)
        return data

    def regular_content(self, title):
        """
        this function is able to regular content
        :param title: news title
        :return: regular title
        """
        title = ''.join(re.findall(u'[\u4e00-\u9fff]+', title))
        title = re.sub('|', '', title)

        return title

    def write_excel(self, file_name, data):
        """
        this function is able to save data to excel
        :param data: data list
        :return:
        """
        book = xlwt.Workbook()
        sheet = book.add_sheet('sheet1')
        c = 0
        for d in data:
            for index in range(len(d)):
                sheet.write(c, index, d[index])
            c += 1
        book.save(file_name)
        print(c, 'save success....')

    def reload_data(self, category):
        """
        this function is reload data when all data excel changed
        :return:
        """
        if category == None:
            print('category is None....')
            return

        garbage_news = self.read_excel(category.all_path, 1)
        self.write_excel(category.garbage_path, garbage_news)
        normal_news = self.read_excel(category.all_path, 0)
        self.write_excel(category.normal_path, normal_news)

    def political_reload_data(self, category):
        """
        this function is reload data when all data excel changed
        :return:
        """
        if category == None:
            print('category is None....')
            return

        garbage_news = self.read_excel(category.all_path, 1)
        self.write_excel(category.garbage_path, garbage_news)
        normal_news = self.read_excel(category.all_path, 0)
        self.write_excel(category.normal_path, normal_news)
        corruption_news = self.read_excel(category.all_path, 2)
        self.write_excel(category.corruption_path, corruption_news)


if __name__ == '__main__':
    dp = DataProcess()
    normal_news = dp.read_excel(r'F:\education\prediction.xls')
    garbage_news = dp.read_excel(r'F:\education\prediction.xls', True)
    dp.write_excel(r'F:\education\garbage_news.xls', garbage_news)
    dp.write_excel(r'F:\education\normal_news.xls', normal_news)