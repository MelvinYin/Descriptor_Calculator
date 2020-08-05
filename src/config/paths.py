import os

ROOT = "/".join(os.path.dirname(__file__).split("/")[:-2])
DATA = os.path.join(ROOT, 'data')
USER_INPUT = os.path.join(DATA, 'input')
USER_OUTPUT = os.path.join(DATA, 'output')
INTERNAL = os.path.join(DATA, 'internal')
DEBUG = os.path.join(DATA, 'debug')
SRC = os.path.join(ROOT, 'src')
INPUT_PDB_INFO = os.path.join(USER_INPUT, 'pdb_info.pkl')
OUTPUT_DESCRS = os.path.join(USER_OUTPUT, 'descrs.pkl')

MATCHERS = os.path.join(INTERNAL, 'matchers')
DESCRS = os.path.join(INTERNAL, 'descrs')

PDB_FILES = os.path.join(INTERNAL, "pdb_files")
PDB_PARSED = os.path.join(INTERNAL, "pdb_files_parsed")

PDB_FILES_SET = set(os.listdir(PDB_FILES))
PDB_PARSED_SET = set(os.listdir(PDB_PARSED))

HB_EXEC = os.path.join(SRC, "parsers", "hb", "hb_calculator")