import sys
import inspect
import os
import pickle
from collections import defaultdict
import heapq
import numpy as np

from config import paths
from pdb_component import pdb_interface
from descr.descr_main import calculate_single

def get_matcher():
    matchers = dict()
    output_matcher = os.path.join(paths.MATCHERS, "efhand_matcher_full.pkl")
    with open(output_matcher, 'rb') as file:
        matchers['ef'] = pickle.load(file)
    return matchers['ef']

def _ui_callback(pdb, cid, seq_marker, matcher):
    return_code, return_val = calculate_single(pdb, cid, seq_marker)

    if return_code == 1:
        return 2, return_val
    if return_code == 2:
        return 3, return_val
    if return_code == 3:
        return 4, return_val
    p_all_sno = matcher.query(return_val)
    return_val_sno = return_val.sno.values
    output = dict()
    output['per_point'] = dict()
    for term, values in p_all_sno.items():
        output['per_point'][return_val_sno[term]] = values

    return 0, output['per_point']

pdb = '1a72'
cid = 'A'
seq_marker = '187'
matcher = get_matcher()

ret_code, output = _ui_callback(pdb, cid, seq_marker, matcher)

score_map = [dict() for __ in range(len(output))]
top_candidates = set()
top_5_scores = np.ones(len(output), dtype=float)


for i, value in enumerate(output.values()):
    for j, (identifier, score) in enumerate(value):
        id_str = (identifier[0], identifier[1], identifier[2])
        if j < 10:
            top_candidates.add(id_str)
        if j < 30:
            top_5_scores[i] = min(top_5_scores[i], score)
        score_map[i][id_str] = score

segment_length = 5
best_score_arr = []

for i in range(len(output) - segment_length):
    max_score, max_cand = 0, None
    for candidate in top_candidates:
        cand_score = 0
        for j in range(segment_length):
            try:
                cand_score += score_map[i+j][candidate]
            except KeyError:
                continue
        if cand_score > max_score:
            max_score = cand_score
            max_cand = candidate
    best_score_arr.append([max_score, max_cand])

best_i = np.argmax([i[0] for i in best_score_arr])
best_candidate = best_score_arr[best_i][1]
# extend on both ends
# to the left
left_i, right_i = best_i, best_i+segment_length
while left_i >= 0:
    if score_map[left_i][best_candidate] < top_5_scores[left_i]:
        break
    left_i -= 1

while right_i < len(output):
    if score_map[right_i][best_candidate] < top_5_scores[right_i]:
        break
    right_i += 1

candidate_output = [None for __ in range(len(output))]
for i in range(left_i, right_i+1):
    candidate_output[i] = [best_candidate, score_map[i][best_candidate],
                           best_score_arr[best_i][0]]

segment_length = 3

while left_i >= segment_length:
    seg_left, seg_right = left_i - segment_length, left_i
    max_score, max_cand = 0, None
    for candidate in top_candidates:
        cand_score = 0
        for j in range(seg_left, seg_right):
            try:
                cand_score += score_map[j][candidate]
            except KeyError:
                continue
        if cand_score > max_score:
            max_score = cand_score
            max_cand = candidate

    while seg_left >= 0:
        if score_map[seg_left][max_cand] < top_5_scores[seg_left]:
            break
        seg_left -= 1
    seg_left = max(seg_left, 0)
    for i in range(seg_left, seg_right):
        candidate_output[i] = [max_cand, score_map[i][max_cand], max_score]
    left_i = seg_left

while right_i <= len(output) - segment_length:
    seg_left, seg_right = right_i, right_i+segment_length
    max_score, max_cand = 0, None
    for candidate in top_candidates:
        cand_score = 0
        for j in range(seg_left, seg_right):
            try:
                cand_score += score_map[j][candidate]
            except KeyError:
                continue
        if cand_score > max_score:
            max_score = cand_score
            max_cand = candidate

    while seg_right < len(output):
        if score_map[seg_right][max_cand] < top_5_scores[seg_right]:
            break
        seg_right += 1

    for i in range(seg_left, seg_right):
        candidate_output[i] = [max_cand, score_map[i][max_cand], max_score]
    right_i = seg_right

for term in candidate_output:
    print(term)