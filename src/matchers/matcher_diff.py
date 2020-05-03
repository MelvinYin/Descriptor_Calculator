import copy
import numpy as np
import pandas as pd

from .ind_matchers_diff.phipsi import PhipsiMatcher
from .ind_matchers_diff.signature import SignatureMatcher
from .matcher_config import Skip

class Matcher:
    def __init__(self):
        self.matchers = dict()

    def load(self, df):
        df_grouped = df.groupby('relative_sno')
        for relative_sno, df_per_sno in df_grouped:
            matcher = MatcherPerSno()
            try:
                matcher.load(df_per_sno)
            except Skip:
                continue
            self.matchers[relative_sno] = matcher
        return

    def query(self, queried):
        df_grouped = queried.groupby('relative_sno')
        p_all_sno = dict()
        for relative_sno, df_per_sno in df_grouped:
            try:
                matcher = self.matchers[relative_sno]
            except KeyError:
                continue
            p_per_sno = matcher.query(df_per_sno)
            p_all_sno[relative_sno] = p_per_sno
        return p_all_sno

class MatcherPerSno:
    # def __init__(self, matchers=(PhipsiMatcher, SignatureMatcher, HbMatcher,
    #                              CovalentMatcher, ContactMatcher)):
    def __init__(self, matchers=(PhipsiMatcher, SignatureMatcher)):
        self.matchers = [x() for x in matchers]
        self.matcher_weights = dict()

    def load(self, df):
        self.identities = list(df[['filename', 'seq_marker', 'cid']].values)
        for i, identity in enumerate(self.identities):
            # cid converted to str as convention, easier to match keys later
            self.identities[i] = tuple(map(str, identity))
        self.identities = np.array(self.identities)
        for matcher in self.matchers:
            matcher.load(df)
        return

    def query(self, datapoint):
        # scorers cannot change order of identity
        weight_p_raw = []
        for matcher in self.matchers:
            weight, p_raw = matcher.query(datapoint)
            assert isinstance(weight, float)
            assert isinstance(p_raw, float)
            assert 0 <= weight <= 1
            weight_p_raw.append([weight, p_raw])
        p_all_seq = self._normalise_prob(weight_p_raw)
        return p_all_seq

    def _normalise_prob(self, weight_p_raw):
        sum_weights = sum([weight_p[0] for weight_p in weight_p_raw])
        p_full = 0
        # ref_length = len(weight_p_raw[0][1])
        for weight, p in weight_p_raw:
            # assert len(p) == ref_length
            weight_norm = weight / sum_weights
            p_norm = p * weight_norm
            p_full += p_norm
        assert p_full <= 1.000001   # float comparison fail otherwise
        assert p_full >= -0.000001
        return p_full



