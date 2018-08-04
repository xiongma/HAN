from tagjieba import dict
class TagJieba(object):
    """ 以 tagjieba 获取 20 个 tag 关键词
    :returns
        ['万达', '王健林', '资产', '广场', '酒店', '项目', '商业', '文旅', '转型', '年会', '集团',
        '股债', '企业', '外界', '投资方', '债务', '数据', '裁员', '模式', '首富']
    """
    def __init__(self):
        import tagjieba
        tagjieba.initialize(dict.slda_tag_dict)

        from tagjieba.analyse import extract_tags, set_stop_words
        set_stop_words(dict.stop_dict)
        self.extract_tags = extract_tags
        self.lcut = tagjieba.lcut

    def top_keys(self, content):
        """
        this function is able to get top K tags which tags property is tag and eng
        :param content: content
        :return: tag and eng tags
        """
        words = self.extract_tags(content, topK=20, allowPOS=['tag', 'eng'])
        return words

    def top_keys_with_all(self, txt):
        words = self.extract_tags(txt, topK=20)
        return words

    def lcut_words(self, txt):
        words = self.lcut(txt)
        return words