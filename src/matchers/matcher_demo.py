import numpy as np
import os
import pickle as pkl

from config import paths
from matchers.matcher import Matcher
import sys
from utils import plots
import pickle
import matplotlib.pyplot as plt

# output_path = os.path.join(paths.ROOT, "GxGxxG_descr.pkl")
# with open(output_path, "rb") as file:
#     descrs = pickle.load(file)
# print(descrs[descrs['sno'] == 12])
# plots.plot_signature_logo(descrs)
# plt.show()
#
#
# with open(os.path.join(paths.ROOT, "efhand_descr.pkl"), "rb") as \
#         pklfile:
#     df = pkl.load(pklfile)
#
# df = df.sort_values(['filename', 'seq_marker', 'cid'])
#
# df_ = df.groupby(['filename', 'seq_marker', 'cid'])
# #
# # # matcher = Matcher()
# # # matcher.load(df)
# #
# with open(os.path.join(paths.ROOT, "efhand_matcher.pkl"), "rb") as \
#         pklfile:
#     matcher = pkl.load(pklfile)

# for key, df_per_file in df_:
#     # print(key)
#     # print(df_per_file)
#     # print("")
#     p_all_sno = matcher.query(df_per_file)
#     # print(p_all_sno)
#
#     for sno, p in p_all_sno.items():
#         print(sno)
#         print(np.array(p))
#     print("")

def load_matchers():
    input_df = os.path.join(paths.DESCRS, "mg_dxdxxd_descr.pkl")
    output_matcher = os.path.join(paths.MATCHERS, "mg_dxdxxd_matcher.pkl")
    with open(input_df, 'rb', -1) as file:
        df = pickle.load(file)
    matcher = Matcher(cropped=False)
    matcher.load(df)
    with open(output_matcher, 'wb') as file:
        pickle.dump(matcher, file, -1)
    print("mg_dxdxxd_descr done")

    input_df = os.path.join(paths.DESCRS, "efhand_descr.pkl")
    output_matcher = os.path.join(paths.MATCHERS, "efhand_matcher.pkl")
    with open(input_df, 'rb', -1) as file:
        df = pickle.load(file)
    matcher = Matcher(cropped=False)
    matcher.load(df)
    with open(output_matcher, 'wb') as file:
        pickle.dump(matcher, file, -1)
    print("efhand_descr done")

    # input_df = os.path.join(paths.DESCRS, "GxGGxG_descr.pkl")
    # output_matcher = os.path.join(paths.MATCHERS, "GxGGxG_matcher.pkl")
    # with open(input_df, 'rb', -1) as file:
    #     df = pickle.load(file)
    # matcher = Matcher(cropped=False)
    # matcher.load(df)
    # with open(output_matcher, 'wb') as file:
    #     pickle.dump(matcher, file, -1)
    # print("GxGGxG_descr done")
    #
    # input_df = os.path.join(paths.DESCRS, "GxxGxG_descr.pkl")
    # output_matcher = os.path.join(paths.MATCHERS, "GxxGxG_matcher.pkl")
    # with open(input_df, 'rb', -1) as file:
    #     df = pickle.load(file)
    # matcher = Matcher(cropped=False)
    # matcher.load(df)
    # with open(output_matcher, 'wb') as file:
    #     pickle.dump(matcher, file, -1)
    # print("GxxGxG_descr done")
    #
    # input_df = os.path.join(paths.DESCRS, "GxGxxG_descr.pkl")
    # output_matcher = os.path.join(paths.MATCHERS, "GxGxxG_matcher.pkl")
    # with open(input_df, 'rb', -1) as file:
    #     df = pickle.load(file)
    # matcher = Matcher(cropped=False)
    # matcher.load(df)
    # with open(output_matcher, 'wb') as file:
    #     pickle.dump(matcher, file, -1)
    # print("GxGxxG_descr done")
    pass

if __name__ == "__main__":
    load_matchers()




