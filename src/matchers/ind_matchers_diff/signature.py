import numpy as np

from utils.blosum import blosum

# Conversion to Enum eventually
# class AA1(Enum):
#     # For p2, enum is not necessarily ordered, so should
#     # add something like this:
#     # __order__ = 'A C D E F G H I K L M N P Q R S T V W Y'
#     A = ['ALA']
#     C = ['CYS']
#     D = ['ASP']
#     E = ['GLU']
#     F = ['PHE']
#     G = ['GLY']
#     H = ['HIS', 'HSE', 'HSD']
#     I = ['ILE']
#     K = ['LYS']
#     L = ['LEU']
#     M = ['MET', 'MSE']
#     N = ['ASN']
#     P = ['PRO']
#     Q = ['GLN']
#     R = ['ARG']
#     S = ['SER']
#     T = ['THR']
#     V = ['VAL']
#     W = ['TRP']
#     Y = ['TYR']

_aa_index = [('ALA', 'A'),
             ('CYS', 'C'),
             ('ASP', 'D'),
             ('GLU', 'E'),
             ('PHE', 'F'),
             ('GLY', 'G'),
             ('HIS', 'H'),
             ('HSE', 'H'),
             ('HSD', 'H'),
             ('ILE', 'I'),
             ('LYS', 'K'),
             ('LEU', 'L'),
             ('MET', 'M'),
             ('MSE', 'M'),
             ('ASN', 'N'),
             ('PRO', 'P'),
             ('GLN', 'Q'),
             ('ARG', 'R'),
             ('SER', 'S'),
             ('THR', 'T'),
             ('VAL', 'V'),
             ('TRP', 'W'),
             ('TYR', 'Y')]

AA3_TO_AA1 = dict(_aa_index)


class SignatureMatcher:
    def __init__(self):
        self.matcher = _SignatureMatcher()

    def load(self, df):
        res = list(df.res.values)
        for i in range(len(res)):
            res[i] = AA3_TO_AA1[res[i]]
        self.matcher.load(res)
        return

    def query(self, df):
        res = df.res.values
        assert len(res) == 1
        res = AA3_TO_AA1[res[0]]
        weight, p = self.matcher.query(res)
        return weight, p

class _SignatureMatcher:
    def __init__(self):
        self.res_distribution = None

    def load(self, res):
        self.res = res
        __, counts = np.unique(res, return_counts=True)
        counts = np.sort(counts)[::-1]
        self.weight = sum(counts[:2]) / sum(counts)
        return

    def query(self, res):
        # Consider substitutability of res
        terms, counts = np.unique(self.res, return_counts=True)
        score = 0
        for term, count in zip(terms, counts):
            p = 1 if term == res else blosum[frozenset([res, term])]
            score += p * count / np.sum(counts)
        assert score <= 1
        return self.weight, score

