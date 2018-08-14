from tagjieba import dict

class TagJieba(object):

    def __init__(self):
        import tagjieba
        tagjieba.initialize(dict.slda_tag_dict)

        from tagjieba.analyse import extract_tags, set_stop_words
        set_stop_words(dict.stop_dict)
        self.extract_tags = extract_tags
        self.lcut = tagjieba.lcut

        # init stop words
        all_stop_words = open(dict.stop_dict, mode='r', encoding='utf-8')
        all_stop_words_ = []
        for word in all_stop_words:
            all_stop_words_.append(word.replace('\n', ''))

        self.stop_words = list(set(all_stop_words_))

    def cut(self, content, cut_all=False, delete_stop_words=False):
        """
        this function is able to cut content, default accurate model
        :param content: content
        :param cut_all: split content model
        :param delete_stop_words: whether delete stop words
        :return: words by cut
        """
        all_words = self.lcut(content, cut_all)
        if delete_stop_words:
            words = [word for word in all_words if word not in self.stop_words]

            return words
        else:
            return all_words