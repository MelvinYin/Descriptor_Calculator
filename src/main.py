import logging
import os
import pickle
import shutil
from time import time

from config import paths
from descr import descr_main
from utils import logs, plots, generic

import matplotlib.pyplot as plt

def main(input_path, output_path):
    """
    Input: motif_pos_pkl
    output: descr_file.pkl
    """

    # todo: if pdb_file is empty (0 bytes), for some reason load_pdb_data
    #  does not throw exception
    logs.set_logging_level()

    # paths
    store_dir = os.path.join(paths.ROOT, 'data', 'store')

    if os.path.isdir(store_dir):
        logging.warning("Store dir exists, deleting.")
        shutil.rmtree(store_dir)
    os.mkdir(store_dir)

    motif_pos_path = input_path
    with open(motif_pos_path, 'rb') as file:
        motif_pos_map = pickle.load(file)
    print(list(motif_pos_map.keys()))
    timecheck = time()
    descrs = descr_main.calculate(motif_pos_map)
    print(f"Time taken: {time() - timecheck}")
    logging.debug(f"Time taken: {time() - timecheck}")
    # for __, descr in descrs.groupby(['filename', 'cid', 'seq_marker']):
    #     calc_descr.write_descr(descr)

    generic.warn_if_exist(paths.OUTPUT_DESCRS)
    # Switching back to pkl to avoid false float comparison failures.
    # with open(os.path.join(paths.ROOT, "final_descr_output_orig.pkl"),
    import numpy as np
    print(descrs.columns)
    print(np.unique(descrs['filename'], return_counts=True))
    with open(output_path, "wb") as file:
        pickle.dump(descrs, file, -1)

    # with open(output_path, "rb") as file:
    #     descrs = pickle.load(file)
    # print(descrs)
    # plots.plot_signature_logo(descrs)
    # plt.show()

def main_with_ui():
    pass


if __name__ == "__main__":
    # input_path = os.path.join(paths.USER_INPUT, "mg_dxdxxd_motif_pos.txt")
    # output_path = os.path.join(paths.DESCRS, "mg_dxdxxd_descr.pkl")
    # main(input_path, output_path)
    #
    # input_path = os.path.join(paths.USER_INPUT, "efhand_motif_pos.txt")
    # output_path = os.path.join(paths.DESCRS, "efhand_descr.pkl")
    # main(input_path, output_path)
    #
    # input_path = os.path.join(paths.USER_INPUT, "GxGGxG_motif_pos.txt")
    # output_path = os.path.join(paths.DESCRS, "GxGGxG_descr.pkl")
    # main(input_path, output_path)
    #
    # input_path = os.path.join(paths.USER_INPUT, "GxxGxG_motif_pos.txt")
    # output_path = os.path.join(paths.DESCRS, "GxxGxG_descr.pkl")
    # main(input_path, output_path)
    #
    # input_path = os.path.join(paths.USER_INPUT, "GxGxxG_motif_pos.txt")
    # output_path = os.path.join(paths.DESCRS, "GxGxxG_descr.pkl")
    # main(input_path, output_path)
    input_path = os.path.join(paths.ROOT, "output.pkl")
    output_path = os.path.join(paths.DESCRS, "output_descr.pkl")
    main(input_path, output_path)
