import numpy as np
import os
import pickle as pkl

from config import paths
from matchers.matcher_diff import Matcher
import sys
from utils import plots
import pickle
import matplotlib.pyplot as plt
from descr.descr_main import calculate_single


# input_df = os.path.join(paths.DESCRS, "mg_dxdxxd_descr.pkl")
# with open(input_df, 'rb', -1) as file:
#     df = pickle.load(file)
# print("opened")
#
# matcher = Matcher()
# matcher.load(df)
#
# for i in range(10):
#     pdb = '1a29'
#     cid = 'A'
#     seq_marker = 12
#
#     return_code, return_val = calculate_single(pdb, cid, seq_marker)
#     if return_code != 0:
#         print(return_code)
#         print(return_val)
#         import sys
#         sys.exit()
#     p_all_sno = matcher.query(return_val)


def load_matchers():
    input_df = os.path.join(paths.DESCRS, "mg_dxdxxd_descr.pkl")
    output_matcher = os.path.join(paths.MATCHERS,
                                  "mg_dxdxxd_matcher_generic.pkl")
    with open(input_df, 'rb', -1) as file:
        df = pickle.load(file)
    matcher = Matcher()
    matcher.load(df)
    with open(output_matcher, 'wb') as file:
        pickle.dump(matcher, file, -1)

    input_df = os.path.join(paths.DESCRS, "efhand_descr.pkl")
    output_matcher = os.path.join(paths.MATCHERS, "efhand_matcher_generic.pkl")
    with open(input_df, 'rb', -1) as file:
        df = pickle.load(file)
    matcher = Matcher()
    matcher.load(df)
    with open(output_matcher, 'wb') as file:
        pickle.dump(matcher, file, -1)

    input_df = os.path.join(paths.DESCRS, "GxGGxG_descr.pkl")
    output_matcher = os.path.join(paths.MATCHERS, "GxGGxG_matcher_generic.pkl")
    with open(input_df, 'rb', -1) as file:
        df = pickle.load(file)
    matcher = Matcher()
    matcher.load(df)
    with open(output_matcher, 'wb') as file:
        pickle.dump(matcher, file, -1)

    input_df = os.path.join(paths.DESCRS, "GxxGxG_descr.pkl")
    output_matcher = os.path.join(paths.MATCHERS, "GxxGxG_matcher_generic.pkl")
    with open(input_df, 'rb', -1) as file:
        df = pickle.load(file)
    matcher = Matcher()
    matcher.load(df)
    with open(output_matcher, 'wb') as file:
        pickle.dump(matcher, file, -1)

    input_df = os.path.join(paths.DESCRS, "GxGxxG_descr.pkl")
    output_matcher = os.path.join(paths.MATCHERS, "GxGxxG_matcher_generic.pkl")
    with open(input_df, 'rb', -1) as file:
        df = pickle.load(file)
    matcher = Matcher()
    matcher.load(df)
    with open(output_matcher, 'wb') as file:
        pickle.dump(matcher, file, -1)
    pass

load_matchers()
