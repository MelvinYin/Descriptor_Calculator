import logging
import os
import pickle
import shutil
from time import time

from config import paths
from descr import descr_main
from utils import logs, plots, generic

def build_descr(input_path, output_path):
    logs.set_logging_level()

    store_dir = os.path.join(paths.ROOT, 'data', 'store')

    if os.path.isdir(store_dir):
        logging.warning("Store dir exists, deleting.")
        shutil.rmtree(store_dir)
    os.mkdir(store_dir)

    motif_pos_path = input_path
    with open(motif_pos_path, 'rb') as file:
        motif_pos_map = pickle.load(file)
    timecheck = time()
    descrs = descr_main.calculate(motif_pos_map)
    print(f"Time taken: {time() - timecheck}")
    logging.debug(f"Time taken: {time() - timecheck}")

    generic.warn_if_exist(paths.OUTPUT_DESCRS)
    # Switching back to pkl to avoid false float comparison failures.
    # with open(os.path.join(paths.ROOT, "final_descr_output_orig.pkl"),

    with open(output_path, "wb") as file:
        pickle.dump(descrs, file, -1)


if __name__ == "__main__":
    input_path = os.path.join(paths.USER_INPUT, "efhand_motif_pos2.pkl")
    output_path = os.path.join(paths.DESCRS, "efhand_descr2.pkl")
    build_descr(input_path, output_path)
    input_path = os.path.join(paths.USER_INPUT, "mg_motif_pos2.pkl")
    output_path = os.path.join(paths.DESCRS, "mg_descr2.pkl")
    build_descr(input_path, output_path)

    input_path = os.path.join(paths.USER_INPUT, "GxGxxG_motif_pos2.pkl")
    output_path = os.path.join(paths.DESCRS, "GxGxxG_descr2.pkl")
    build_descr(input_path, output_path)

    # input_path = os.path.join(paths.USER_INPUT, "GxGxxG_motif_pos2.pkl")
    # output_path = os.path.join(paths.DESCRS, "GxGxxG_descr2.pkl")
    # build_descr(input_path, output_path)
    #
    # input_path = os.path.join(paths.USER_INPUT, "GxGGxG_motif_pos2.pkl")
    # output_path = os.path.join(paths.DESCRS, "GxGGxG_descr2.pkl")
    # build_descr(input_path, output_path)

    # input_path = os.path.join(paths.USER_INPUT, "GxxGxG_motif_pos.pkl")
    # output_path = os.path.join(paths.DESCRS, "GxxGxG_descr.pkl")
    # build_descr(input_path, output_path)
    #
    # input_path = os.path.join(paths.USER_INPUT, "GxGxxG_motif_pos.pkl")
    # output_path = os.path.join(paths.DESCRS, "GxGxxG_descr.pkl")
    # build_descr(input_path, output_path)
    #
    # input_path = os.path.join(paths.USER_INPUT, "GxGGxG_motif_pos.pkl")
    # output_path = os.path.join(paths.DESCRS, "GxGGxG_descr.pkl")
    # build_descr(input_path, output_path)