from os.path import split as __pathSplit, realpath as __realpath
__real_path = __pathSplit(__realpath(__file__))[0]

# default dictionary
tag_dict = __real_path + '/tag_dict.txt'
# # add lda dictionary
# tag_lda_dict = __real_path + '/jb_slda.txt'
# ida regional unloading in mysql
ida_region = __real_path + '/ida_region.json'
# stop words dictionary
stop_dict = __real_path + '/stop.txt'
# combine slda and tags dictionary
slda_tag_dict = __real_path + '/mix_lda_tag.txt'
# tags id
tag_word_ids = __real_path + '/tagWord2id.json'