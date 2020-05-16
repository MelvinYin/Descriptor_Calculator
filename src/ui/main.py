import matplotlib.pyplot as plt
# import matplotlib
#
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import sys
import inspect
src_path = inspect.currentframe().f_code.co_filename.rsplit("/", maxsplit=2)[0]
sys.path.append(src_path)
from bokeh.plotting import figure, show, ColumnDataSource
from bokeh.models.widgets import Div, RadioButtonGroup, Select, Button, \
    HTMLTemplateFormatter
from bokeh.models.widgets import DataTable, DateFormatter, TableColumn
import os
import pickle
from bokeh.models.widgets import TextInput
from bokeh.plotting import show, curdoc
from bokeh.layouts import row, column
from collections import defaultdict
import heapq
from bokeh.palettes import Category10
from config import paths
from pdb_component import pdb_interface
from descr.descr_main import calculate_single
import numpy as np
from collections import OrderedDict

from bokeh.plotting import figure, show, ColumnDataSource, curdoc
from bokeh.models import DataRange1d, HoverTool, BoxZoomTool, PanTool, \
    WheelZoomTool, ResetTool, UndoTool, Plot, Text, Range1d, ImageURL
from bokeh.models.tickers import FixedTicker

import inspect
from enum import Enum
from bokeh.layouts import row, column
from bokeh.models.widgets import Paragraph, Div
from bokeh.models.layouts import WidgetBox, Spacer
from bokeh.models import DataTable, TableColumn, ColumnDataSource


class SegmentCalculator:
    def __init__(self, segment_length=7, extension_length=5,
                 num_top_candidates=10, extension_cutoff=50):
        self.segment_len = segment_length
        self.extension = extension_length
        self.matchers = self._load_matchers()
        self.num_top_candidates = num_top_candidates
        self.extension_cutoff = extension_cutoff

    def _load_matchers(self):
        matchers = dict()
        output_matcher = os.path.join(paths.MATCHERS,
                                      "efhand_matcher.pkl")
        with open(output_matcher, 'rb') as file:
            matchers['ef'] = pickle.load(file)

        output_matcher = os.path.join(paths.MATCHERS,
                                      "mg_dxdxxd_matcher.pkl")
        with open(output_matcher, 'rb') as file:
            matchers['mg'] = pickle.load(file)

        output_matcher = os.path.join(paths.MATCHERS,
                                      "GxGGxG_matcher.pkl")
        with open(output_matcher, 'rb') as file:
            matchers['gxgxxg'] = pickle.load(file)

        output_matcher = os.path.join(paths.MATCHERS,
                                      "GxxGxG_matcher.pkl")
        with open(output_matcher, 'rb') as file:
            matchers['gxxgxg'] = pickle.load(file)

        output_matcher = os.path.join(paths.MATCHERS,
                                      "GxGxxG_matcher.pkl")
        with open(output_matcher, 'rb') as file:
            matchers['gxggxg'] = pickle.load(file)

        output_matcher = os.path.join(paths.MATCHERS, "output_matcher.pkl")
        with open(output_matcher, 'rb') as file:
            matchers['single'] = pickle.load(file)
        return matchers

    # def query(self, descr, pdb, cid, seq_marker):
    #     matcher = self.matchers[descr]
    #     return_code, return_val = calculate_single(pdb, cid, seq_marker)
    #     p_all_sno = matcher.query(return_val)
    #     sno_list = return_val.sno.values
    #     sno_probs = dict()
    #     for relative_sno, probs in p_all_sno.items():
    #         actual_sno = sno_list[relative_sno]
    #         sno_probs[actual_sno] = probs
    #
    #     descr_length = len(sno_probs)
    #     score_map = [dict() for __ in range(descr_length)]
    #     top_candidates = set()
    #     top_n_scores = np.ones(descr_length, dtype=float)
    #
    #     for i, probs in enumerate(sno_probs.values()):
    #         for j, (identifier, score) in enumerate(probs):
    #             id_str = (identifier[0], identifier[1], identifier[2])
    #             if j < self.num_top_candidates:
    #                 top_candidates.add(id_str)
    #             if j < self.extension_cutoff:
    #                 top_n_scores[i] = min(top_n_scores[i], score)
    #             score_map[i][id_str] = score
    #
    #     best_scores = []
    #
    #     for i in range(descr_length - self.segment_len):
    #         max_score, max_cand = 0, None
    #         for candidate in top_candidates:
    #             cand_score = 0
    #             for j in range(self.segment_len):
    #                 try:
    #                     cand_score += score_map[i + j][candidate]
    #                 except KeyError:
    #                     # effectively skip these candidates
    #                     continue
    #             if cand_score > max_score:
    #                 max_score = cand_score
    #                 max_cand = candidate
    #         best_scores.append([max_score, max_cand])
    #
    #     best_i = np.argmax([i[0] for i in best_scores])
    #     # to dismiss ide inspection error
    #     best_i = int(best_i)
    #     best_candidate = best_scores[best_i][1]
    #     # extend on both ends
    #     # to the left
    #     left_i, right_i = best_i, best_i + self.segment_len
    #     while left_i > 0:
    #         if score_map[left_i][best_candidate] < top_n_scores[left_i]:
    #             break
    #         left_i -= 1
    #
    #     # to the right
    #     while right_i < descr_length - 1:
    #         if score_map[right_i][best_candidate] < top_n_scores[right_i]:
    #             break
    #         right_i += 1
    #
    #     candidate_output = [[] for __ in range(descr_length)]
    #
    #     for i in range(left_i, right_i + 1):
    #         candidate_output[i] = [best_candidate, score_map[i][best_candidate],
    #                                best_scores[best_i][0]]
    #
    #     while left_i >= self.extension:
    #         seg_left, seg_right = left_i - self.extension, left_i
    #         max_score, max_cand = 0, None
    #         for candidate in top_candidates:
    #             cand_score = 0
    #             for j in range(seg_left, seg_right):
    #                 try:
    #                     cand_score += score_map[j][candidate]
    #                 except KeyError:
    #                     continue
    #             if cand_score > max_score:
    #                 max_score = cand_score
    #                 max_cand = candidate
    #         while seg_left >= 0:
    #             if seg_left in score_map and max_cand in score_map[seg_left] \
    #                 and score_map[seg_left][max_cand]< top_n_scores[seg_left]:
    #                 break
    #             seg_left -= 1
    #         seg_left = max(seg_left, 0)
    #         for i in range(seg_left, seg_right):
    #             if max_cand in score_map[i]:
    #                 score_to_add = score_map[i][max_cand]
    #             else:
    #                 score_to_add = 0.01
    #             candidate_output[i] = [max_cand, score_to_add, max_score]
    #         left_i = seg_left
    #
    #     while right_i <= descr_length - self.extension:
    #         seg_left, seg_right = right_i, right_i + self.extension
    #         max_score, max_cand = 0, None
    #         for candidate in top_candidates:
    #             cand_score = 0
    #             for j in range(seg_left, seg_right):
    #                 try:
    #                     cand_score += score_map[j][candidate]
    #                 except KeyError:
    #                     continue
    #             if cand_score > max_score:
    #                 max_score = cand_score
    #                 max_cand = candidate
    #         while seg_right < descr_length:
    #             if score_map[seg_right][max_cand] < top_n_scores[seg_right]:
    #                 break
    #             seg_right += 1
    #
    #         for i in range(seg_left, seg_right):
    #             candidate_output[i] = [max_cand, score_map[i][max_cand], max_score]
    #         right_i = seg_right
    #
    #     mapped_output = dict()
    #     for sno, value in zip(sno_probs.keys(), candidate_output):
    #         mapped_output[sno] = value
    #     return mapped_output

    def query(self, descr, pdb, cid, seq_marker):
        matcher = self.matchers[descr]
        return_code, return_val = calculate_single(pdb, cid, seq_marker)
        p_all_sno = matcher.query(return_val)
        sno_list = return_val.sno.values
        sno_probs = dict()
        for relative_sno, probs in p_all_sno.items():
            actual_sno = sno_list[relative_sno]
            sno_probs[actual_sno] = probs

        descr_length = len(sno_probs)
        score_map = [dict() for __ in range(descr_length)]
        top_candidates = set()
        top_n_scores = np.ones(descr_length, dtype=float)

        for i, probs in enumerate(sno_probs.values()):
            for j, (identifier, score) in enumerate(probs):
                # if identifier[0] not in ("1a5z", "2bi7", "2cul", "1ps9",
                #                          "1jwb", "1lua"):
                #     continue
                id_str = (identifier[0], identifier[1], identifier[2])
                top_candidates.add(id_str)
                top_n_scores[i] = min(top_n_scores[i], score)
                score_map[i][id_str] = score

        best_scores = []
        for i in range(descr_length - self.segment_len):
            max_score, max_cand = 0, None
            for candidate in top_candidates:
                cand_score = 0
                for j in range(self.segment_len):
                    try:
                        cand_score += score_map[i + j][candidate]
                    except KeyError:
                        # effectively skip these candidates
                        continue
                if cand_score > max_score:
                    max_score = cand_score
                    max_cand = candidate
            best_scores.append([max_score, max_cand])
        best_i = np.argmax([i[0] for i in best_scores])
        # to dismiss ide inspection error
        best_i = int(best_i)
        best_candidate = best_scores[best_i][1]
        # extend on both ends
        # to the left

        left_i, right_i = best_i, best_i + self.segment_len
        while left_i > 0:
            if score_map[left_i][best_candidate] < top_n_scores[left_i]:
                break
            left_i -= 1
        # to the right
        while right_i < descr_length - 1:
            if score_map[right_i][best_candidate] < top_n_scores[right_i]:
                break
            right_i += 1
        candidate_output = [[] for __ in range(descr_length)]

        for i in range(left_i, right_i + 1):
            candidate_output[i] = [best_candidate, score_map[i][best_candidate],
                                   best_scores[best_i][0]]

        while left_i >= self.extension:
            seg_left, seg_right = left_i - self.extension, left_i
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
                if seg_left in score_map and max_cand in score_map[seg_left] \
                        and \
                        score_map[seg_left][max_cand] < top_n_scores[seg_left]:
                    break
                seg_left -= 1
            seg_left = max(seg_left, 0)
            for i in range(seg_left, seg_right):
                if max_cand in score_map[i]:
                    score_to_add = score_map[i][max_cand]
                else:
                    score_to_add = 0.01
                candidate_output[i] = [max_cand, score_to_add, max_score]
            left_i = seg_left
        while right_i <= descr_length - self.extension:
            seg_left, seg_right = right_i, right_i + self.extension
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
            while seg_right < descr_length:
                if score_map[seg_right][max_cand] < top_n_scores[seg_right]:
                    break
                seg_right += 1

            for i in range(seg_left, seg_right):
                candidate_output[i] = [max_cand, score_map[i][max_cand],
                                       max_score]
            right_i = seg_right

        mapped_output = dict()
        for sno, value in zip(sno_probs.keys(), candidate_output):
            mapped_output[sno] = value
        return mapped_output

class ButtonComponent:
    def __init__(self, specs, widget_callback):
        self.widget_callback = widget_callback
        self.widget = self._set_button(specs)

    def _set_button(self, specs):
        button = Button()
        button.label = specs['text']
        button.width = specs['width']
        button.height = specs['height']
        button.on_click(self.widget_callback)
        return button

class TextBoxComponent:
    def __init__(self, specs):
        self.figure = self._set_TB(specs)

    def _set_TB(self, specs):
        TB = Div(text=specs['title'])
        TB.width = specs['width']
        TB.height = specs['height']
        if 'style' in specs:
            TB.style = specs['style']
        return TB

class DropDownComponent:
    def __init__(self, specs, callback=None):
        self.specs = specs
        self.figure = self._set_dropdown(callback)

    def _set_dropdown(self, callback):
        dropdown = Select(title=self.specs["title"],
                          value=self.specs['menu'][0][0],
                          options=self.specs['menu'],
                          width=self.specs['width'],
                          height=self.specs['height'])
        if callback:
            dropdown.on_change('value', callback)
        return dropdown

class MatcherManager:
    def __init__(self):
        self.matchers = self._load_matchers()
        self.matchers_generic = self._load_matchers_generic()
        self.segment_calculater = SegmentCalculator()
        self.ui = UI(self._ui_callback)
        self.figure = self.ui.figure

    def _ui_callback(self, descr, pdb, cid, seq_marker):
        if descr not in self.matchers:
            error_msg = f"Exception: {descr} not found in list of matches. " \
                        f"This is an internal error."
            return 1, error_msg
        return_code, return_val = calculate_single(pdb, cid, seq_marker)
        if return_code == 1:
            return 2, return_val
        if return_code == 2:
            return 3, return_val
        if return_code == 3:
            return 4, return_val
        p_all_sno = self.matchers[descr].query(return_val)
        return_val_sno = return_val.sno.values
        residues = return_val.res
        alphabets = []
        AA3_to_AA1 = dict(ALA='A', CYS='C', ASP='D', GLU='E', PHE='F', GLY='G',
                          HIS='H', HSE='H', HSD='H', ILE='I', LYS='K', LEU='L',
                          MET='M', MSE='M', ASN='N', PRO='P', GLN='Q', ARG='R',
                          SER='S', THR='T', VAL='V', TRP='W', TYR='Y')
        for res in residues:
            if res in AA3_to_AA1:
                alphabets.append(AA3_to_AA1[res])
            else:
                alphabets.append("X")
        print(alphabets)
        output = dict()
        output['per_point'] = dict()
        for term, values in p_all_sno.items():
            output['per_point'][return_val_sno[term]] = values

        # return_code, return_val = calculate_single(pdb, cid, seq_marker)
        if return_code != 0:
            print(f"in _ui_callback, return_code is {return_code}")
            raise Exception
        output['generic'] = defaultdict(dict)
        for descr_local, matcher in self.matchers_generic.items():
            p_all_sno = matcher.query(return_val)
            return_val_sno = return_val.sno.values
            for term, values in p_all_sno.items():
                output['generic'][descr_local][return_val_sno[term]] = values

        segmented_scores = self.segment_calculater.query(descr, pdb, cid,
                                                     seq_marker)
        output['segment'] = segmented_scores
        return 0, output

    def _load_matchers_generic(self):
        matchers = dict()
        output_matcher = os.path.join(paths.MATCHERS, "efhand_matcher_generic.pkl")
        with open(output_matcher, 'rb') as file:
            matchers['ef'] = pickle.load(file)

        output_matcher = os.path.join(paths.MATCHERS, "mg_dxdxxd_matcher_generic.pkl")
        with open(output_matcher, 'rb') as file:
            matchers['mg'] = pickle.load(file)

        output_matcher = os.path.join(paths.MATCHERS, "GxGGxG_matcher_generic.pkl")
        with open(output_matcher, 'rb') as file:
            matchers['gxggxg'] = pickle.load(file)

        output_matcher = os.path.join(paths.MATCHERS, "GxxGxG_matcher_generic.pkl")
        with open(output_matcher, 'rb') as file:
            matchers['gxxgxg'] = pickle.load(file)

        output_matcher = os.path.join(paths.MATCHERS, "GxGxxG_matcher_generic.pkl")
        with open(output_matcher, 'rb') as file:
            matchers['gxgxxg'] = pickle.load(file)
        return matchers


    def _load_matchers(self):
        matchers = dict()
        output_matcher = os.path.join(paths.MATCHERS, "efhand_matcher.pkl")
        with open(output_matcher, 'rb') as file:
            matchers['ef'] = pickle.load(file)

        output_matcher = os.path.join(paths.MATCHERS, "mg_dxdxxd_matcher.pkl")
        with open(output_matcher, 'rb') as file:
            matchers['mg'] = pickle.load(file)

        output_matcher = os.path.join(paths.MATCHERS, "GxGGxG_matcher.pkl")
        with open(output_matcher, 'rb') as file:
            matchers['gxgxxg'] = pickle.load(file)

        output_matcher = os.path.join(paths.MATCHERS, "GxxGxG_matcher.pkl")
        with open(output_matcher, 'rb') as file:
            matchers['gxxgxg'] = pickle.load(file)

        output_matcher = os.path.join(paths.MATCHERS, "GxGxxG_matcher.pkl")
        with open(output_matcher, 'rb') as file:
            matchers['gxggxg'] = pickle.load(file)

        output_matcher = os.path.join(paths.MATCHERS, "output_matcher.pkl")
        with open(output_matcher, 'rb') as file:
            matchers['single'] = pickle.load(file)
        return matchers


class IndividualFigure:
    def __init__(self, descr_length=30):
        self.descr_length = descr_length
        self.url_descr_map = self._set_url_descr_map()
        self.curr_descr = None
        self.CDS = self._create_initial_CDS()
        self.line_CDS = self._create_initial_line_CDS()
        self.plot_legend_CDS = self._init_plot_legend_CDS()
        self.legend_table_header = self._build_plot_legend_table_header()
        self.legend_table = self._build_plot_legend_table()
        # self.tick_label_map = self._get_initial_ticks_label_mapping()
        self.figure, self.fig_x_range, self.fig_x_ticker = \
            self._set_initial_figure()
        self.logo_CDS = self._set_logo_CDS()
        self.logo = self._set_logo()
        self.full_figure = self._build_full_fig()

    def _set_url_descr_map(self):
        mapping = dict()
        url = os.path.join(os.path.basename(os.path.dirname(__file__)),
                           "static", "ef_hand_logo.png")
        mapping['ef'] = url
        url = os.path.join(os.path.basename(os.path.dirname(__file__)),
                           "static", "init_logo.png")
        mapping['init'] = url
        url = os.path.join(os.path.basename(os.path.dirname(__file__)),
                           "static", "gxggxg_logo.png")
        mapping['gxggxg'] = url
        url = os.path.join(os.path.basename(os.path.dirname(__file__)),
                           "static", "gxgxxg_logo.png")
        mapping['gxgxxg'] = url
        url = os.path.join(os.path.basename(os.path.dirname(__file__)),
                           "static", "gxxgxg_logo.png")
        mapping['gxxgxg'] = url
        url = os.path.join(os.path.basename(os.path.dirname(__file__)),
                           "static", "mg_logo.png")
        mapping['mg'] = url

        url = os.path.join(os.path.basename(os.path.dirname(__file__)),
                           "static", "gxgxxg_logo.png")
        mapping['single'] = url
        return mapping

    def _set_logo_CDS(self):
        data = dict(url=[self.url_descr_map['init']], x=[0])
        source = ColumnDataSource(data)
        return source


    def _build_plot_legend_table(self):
        template = """
        <div style="background:<%=color%>; color:<%=color%>;">
        <%= value %></div>
        """
        formatter = HTMLTemplateFormatter(template=template)
        color_width = 70
        pdb_width = 60
        cid_width = 60
        resno_width = 130
        pos_width = 100
        total_width = color_width + pdb_width + cid_width + resno_width + pos_width

        columns = [TableColumn(field="color", title="Line Colour", width=color_width,
                               formatter=formatter),
                   TableColumn(field="pdb", title="PDB Code", width=pdb_width),
                   TableColumn(field="cid", title="Chain ID", width=cid_width),
                   TableColumn(field="resno", title="Starting Residue No",
                               width=resno_width),
                   TableColumn(field="seqno", title="Position Range",
                               width=pos_width)]
        data_table = DataTable(source=self.plot_legend_CDS, columns=columns,
                               width=total_width, height=150, index_position=None,
                               fit_columns=True)
        return data_table

    def _build_plot_legend_table_header(self):
        specs = dict(title="Matching Structures", height=7, width=500)
        header = TextBoxComponent(specs)
        return header

    def _init_plot_legend_CDS(self):
        data = dict(color=["#FFFFFF" for __ in range(10)],
                    pdb=["" for __ in range(10)], cid=["" for __ in range(10)],
                    resno=["" for __ in range(10)],
                    seqno=["" for __ in range(10)])
        source = ColumnDataSource(data)
        return source

    def _create_initial_line_CDS(self):
        initial_data = dict()
        initial_data['xs_line'] = [[0] for __ in range(10)]
        initial_data['ys_line'] = [[0] for __ in range(10)]
        initial_data['color_line'] = ['#1f77b4' for __ in range(10)]
        CDS = ColumnDataSource(initial_data)
        return CDS

    def _create_initial_CDS(self):
        """
        Dict sent to CDS should have list instead of tuple as values, otherwise
        CDS.stream will fail.
        """
        # add test? to make sure all values used in this class is in init_data
        initial_data = dict()
        initial_data['xs'] = list(range(self.descr_length))
        initial_data['ys'] = list([0 for i in range(self.descr_length)])
        initial_data['color'] = list(['#1f77b4' for i in range(self.descr_length)])
        CDS = ColumnDataSource(initial_data)
        return CDS

    def _set_logo(self):
        boxzoom = BoxZoomTool()
        pan = PanTool(dimensions='width')
        wheelzoom = WheelZoomTool(dimensions='width')
        reset = ResetTool()
        undo = UndoTool()
        tools = [pan, boxzoom, wheelzoom, reset, undo]
        p = figure(width=1000, height=200, tools=tools)
        glyph = ImageURL(url="url", x='x', y=4.3, w=30, h=4.4)
        p.add_glyph(self.logo_CDS, glyph)
        p.x_range = self.fig_x_range
        p.y_range = Range1d(start=0, end=4.4)
        p.toolbar.active_drag = pan
        p.toolbar.logo = None
        p.xaxis.ticker = self.fig_x_ticker
        p.xaxis.axis_label = 'Residue Position'
        p.yaxis.axis_label = 'Information Bits'
        return p

    def _set_initial_figure(self):
        boxzoom = BoxZoomTool()
        pan = PanTool(dimensions='width')
        wheelzoom = WheelZoomTool(dimensions='width')
        reset = ResetTool()
        undo = UndoTool()

        tools = [pan, boxzoom, wheelzoom, reset, undo]

        p = figure(plot_width=1000, plot_height=300, tools=tools)
        p.toolbar.active_drag = pan
        p.toolbar.active_scroll = wheelzoom
        p.toolbar.active_tap = None
        p.toolbar.logo = None

        x_range_obj = Range1d(start=0, end=25)

        p.x_range = x_range_obj
        p.y_range = Range1d(start=-0.1, end=1.1)

        initial_ticks = list(range(30))
        x_ticker = FixedTicker(ticks=initial_ticks)
        p.xaxis.ticker = x_ticker
        p.xgrid.ticker = x_ticker

        initial_ticks = list([i/10 for i in range(11)])
        y_ticker = FixedTicker(ticks=initial_ticks)
        p.yaxis.ticker = y_ticker
        p.ygrid.ticker = y_ticker

        p.title.text = 'Segment Matching Of Structures'
        p.title.align = 'center'

        p.yaxis.axis_label = 'Score'
        p.x("xs", 'ys', source=self.CDS, line_color='color',
            line_dash='solid', line_width=2)
        p.multi_line(xs="xs_line", ys='ys_line', color='color_line',
               source=self.line_CDS, line_width=2)
        return p, x_range_obj, x_ticker

    def _build_full_fig(self):
        segment_legend_col = column(
            [self.legend_table_header.figure, self.legend_table])
        full_fig = column([segment_legend_col, self.figure, self.logo])
        return full_fig

    def _update_xaxis(self, start, end):
        ticker = FixedTicker(ticks=list(range(start, end+1)))
        self.figure.xaxis.ticker = ticker
        self.figure.xgrid.ticker = ticker
        self.figure.x_range.update(start=start-1, end=start+ 25)
        return ticker

    def figure_update(self, data):
        self.curr_descr = data['descr']
        patch = defaultdict(list)
        seen_pdbs = OrderedDict()
        xs_update = [data['xs'][i] for i in range(self.descr_length)]
        ys_update = [data['ys'][i] for i in range(self.descr_length)]
        for i in range(self.descr_length):
            if data['name'][i] is None:
                continue
            id_value = tuple(data['name'][i])
            if id_value in seen_pdbs:
                seen_pdbs[id_value] += 1
            else:
                seen_pdbs[id_value] = 1
            indice = min([len(seen_pdbs) - 1, 9])
            assert indice < len(Category10[10]), indice
            patch['color'].append(Category10[10][indice])
            patch['xs'].append(xs_update[i])
            patch['ys'].append(ys_update[i])

        patch_line = defaultdict(list)
        patch_legend = defaultdict(list)
        count = 0
        for i, (key, value) in enumerate(seen_pdbs.items()):
            patch_line['color_line'].append(Category10[10][i])
            patch_line['xs_line'].append(patch['xs'][count:count+value])
            patch_line['ys_line'].append(patch['ys'][count:count + value])

            patch_legend['color'].append(Category10[10][i])
            assert len(key) == 3
            patch_legend['pdb'].append(key[0])
            patch_legend['resno'].append(key[1])
            patch_legend['cid'].append(key[2])
            patch_legend['seqno'].append(f"{patch['xs'][count]} - "
                                         f"{patch['xs'][count+value-1]}")
            count += value
            if i == 9:
                break

        self.line_CDS.data = patch_line
        self.CDS.data = patch
        self.plot_legend_CDS.data = patch_legend
        x_ticker = self._update_xaxis(data['xs'][0], data['xs'][-1])

        self.logo_CDS.data['url'] = [self.url_descr_map[self.curr_descr]]
        self.logo_CDS.data['x'] = [data['xs'][0] - 0.5]
        self.logo.xaxis.ticker = x_ticker
        return True


class UI:
    def __init__(self, ui_callback):
        self.descr_value = 'ef'
        self.pdb_value = '1a72'
        self.cid_value = 'A'
        self.seq_start_value = '187'
        self.ui_callback = ui_callback

        self.descr_key_title_map = self._build_descr_key_title_map()
        self.result_CDS = self._init_result_CDS()
        self.summary_CDS = self._init_summary_CDS()
        self.descr_match_CDS = self._init_descr_match_CDS()
        self.all_descr_CDS = self._init_all_descr_CDS()


        self.console_header = self._build_console_header()
        self.console = self._build_console()

        self.all_descr_header = self._build_all_descr_header()
        self.all_descr_table = self._build_all_descr_table()

        self.msg_board = self._build_msg_board()
        self.project_header = self._build_p_header()
        self.project_description = self._build_p_descr()
        self.pdb_ti = self._build_pdb_ti()
        self.cid_ti = self._build_cid_ti()
        self.seq_start_ti = self._build_seq_start_ti()
        self.enter_button = self._build_enter_button()
        self.descr_dropdown = self._build_descr_dropdown()
        self.result_table_header = self._build_result_table_header()
        self.summary_table_header = self._build_summary_table_header()
        self.descr_match_table_header = self._build_descr_match_table_header()
        self.segment_header = self._build_segment_header()

        self.summary_table = self._build_summary_table()
        self.result_table = self._build_result_table()
        self.segment_plot = self._build_segment_plot()
        self.descr_match_table = self._build_descr_match_table()

        self.figure = self._build_figure()

    def _set_descr_value(self, attr, old, new):
        self.descr_value = new

    def _set_pdb_value(self, attr, old, new):
        self.pdb_value = new

    def _set_cid_value(self, attr, old, new):
        self.cid_value = new

    def _set_seq_start_value(self, attr, old, new):
        self.seq_start_value = new

    def add_to_console(self, message):
        self.console.figure.text += message

    def set_result_callback(self, *args):
        print("pre_ui_callback")
        if self.descr_value is None or self.pdb_value is None or \
                self.cid_value is None or self.seq_start_value is None:
            e_msg = f"<br>One of the input values is None, please try again.<br>"
            self.add_to_console(e_msg)
            return
        msg = f"<br>Called with args descr_value: [{self.descr_value}], " \
              f"pdb_value: [{self.pdb_value}], cid_value: [{self.cid_value}]," \
              f" seq_start_value: [{self.seq_start_value}].<br>"
        print(msg)
        self.add_to_console(msg)
        ret_code, results = self.ui_callback(self.descr_value, self.pdb_value,
                                    self.cid_value, self.seq_start_value)
        print("post_ui_callback")
        if ret_code in (1, 2, 3, 4):
            self.add_to_console(results)
            return
        self.add_to_console("Success.<br>")
        cds_patch = defaultdict(list)

        best_match = defaultdict(int)
        track_first_resno = dict()
        p_per_point = results['per_point']

        if len(p_per_point) != 30:
            print(f"in set_result_callback, len(p_per_point) != 30, is {len(p_per_point)}")
            raise Exception

        for i, (sno, p) in enumerate(p_per_point.items()):
            _matches = []
            for j, term in enumerate(p[:3]):
                if len(term) != 2:
                    print(f"in set_result_callback, len(term) != 2, "
                        f"is {term}")
                    raise Exception
                if len(term[0]) != 3:
                    print(f"in set_result_callback, len(term[0]) != 3, "
                          f"is {term[0]}")
                    raise Exception
                index = i * 3 + j
                pdb_id, start_resno, cid = term[0]
                match_value = term[1]
                cds_patch['resno'].append((index, sno))
                cds_patch['pdb'].append((index, pdb_id))
                cds_patch['cid'].append((index, cid))
                cds_patch['match_resno'].append((index, start_resno))
                cds_patch['match_vals'].append((index, "{0:.3f}".format(
                    match_value)))
                _matches.append([pdb_id, cid, match_value])
                if (pdb_id, cid) in track_first_resno:
                    track_first_resno[(pdb_id, cid)] = min(
                        track_first_resno[(pdb_id, cid)], start_resno)
                else:
                    track_first_resno[(pdb_id, cid)] = start_resno
            summed_match_val = sum(i[2] for i in _matches)
            for match in _matches:
                best_match[(match[0], match[1])] += match[2] / summed_match_val
        self.result_CDS.patch(cds_patch)

        top_5 = []
        for key, value in best_match.items():
            heapq.heappush(top_5, (-value, key))

        summary_patch = defaultdict(list)
        for i in range(5):
            if top_5:
                match_value, (pdb_id, cid) = heapq.heappop(top_5)
                match_value = match_value * (-1)
                summary_patch['pdb'].append((i, pdb_id))
                summary_patch['cid'].append((i, cid))
                summary_patch['match_resno'].append(
                    (i, track_first_resno[(pdb_id, cid)]))
                summary_patch['match_vals'].append(
                    (i, "{0:.3f}".format(match_value)))
            else:
                summary_patch['pdb'].append((i, ""))
                summary_patch['cid'].append((i, ""))
                summary_patch['match_resno'].append((i, ""))
                summary_patch['match_vals'].append((i, ""))
        self.summary_CDS.patch(summary_patch)


        if self.descr_value not in results['generic']:
            print(f"WARNING: self.descr_value not in results['generic'], "
                  f"{self.descr_value}")
        else:
            p_generic = results['generic'][self.descr_value]
            descr_match_patch = defaultdict(list)
            summed_descr_score = sum(p_generic.values())
            descr_match_patch['resno'].append((0, "Total"))
            descr_match_patch['match_vals'].append(
                (0, "{0:.3f}".format(summed_descr_score)))
            for i, (sno, p) in enumerate(p_generic.items()):
                i += 1
                descr_match_patch['resno'].append((i, sno))
                descr_match_patch['match_vals'].append(
                    (i, "{0:.3f}".format(p)))
            self.descr_match_CDS.patch(descr_match_patch)
            all_descr_patch = defaultdict(list)
            scores = []
            for i, (descr, values) in enumerate(results['generic'].items()):
                summed_descr_score = sum(values.values())
                scores.append((self.descr_key_title_map[descr], summed_descr_score))

            sorted_i = np.argsort([i[1] for i in scores])[::-1]
            for i in range(len(sorted_i)):
                selected_i = sorted_i[i]
                all_descr_patch['descr'].append((i, scores[selected_i][0]))
                all_descr_patch['score'].append((i, "{0:.3f}".format(scores[
                                                                         selected_i][1])))
            self.all_descr_CDS.patch(all_descr_patch)

        segment_scores = results['segment']
        segment_plot_patch = defaultdict(list)

        for i, (sno, values) in enumerate(segment_scores.items()):
            segment_plot_patch['xs'].append(sno)

            if not values:
                segment_plot_patch['ys'].append(0)
                segment_plot_patch['name'].append(None)
            else:
                protein_id, score_1, score_2 = values
                segment_plot_patch['ys'].append(score_1)
                segment_plot_patch['name'].append(protein_id)
        segment_plot_patch['descr'] = self.descr_value
        self.segment_plot.figure_update(segment_plot_patch)
        return

    def _build_msg_board(self):
        specs = dict(title="Values to try:<br><br>EF_Hand<br><br>"
                           "pdb_code: 1a29<br>residue: 12<br> chain: A<br><br>"
                           "pdb_code: 1avs<br>residue: 22<br> chain: B<br><br>"
                           "pdb_code: 1aui<br>residue: 22<br> chain: B<br><br>"
                           "<br>MG_DxDxDG<br><br>"
                           "pdb_code: 117e<br>residue: 1104<br> chain: B<br><br>"
                           "pdb_code: 1mjz<br>residue: 54<br> chain: A<br><br>"
                           "pdb_code: 1ipw<br>residue: 54<br> chain: B<br><br>"
                           "<br>GxGxxG<br><br>"
                           "pdb_code: 1a71<br>residue: 187<br> chain: B<br><br>"
                           "pdb_code: 1a72<br>residue: 187<br> chain: A<br><br>"
                           "<br>GxxGxG<br><br>"
                           "pdb_code: 121p<br>residue: 2<br> chain: A<br><br>"
                           , height=50, width=500)
        descr = TextBoxComponent(specs)
        return descr

    def _build_console(self):
        style = {'overflow-y': 'scroll', 'height': '300px', 'width': '500px'}
        specs = dict(title="Application is live.<br><br>",
                     height=300, width=500, style=style)
        console = TextBoxComponent(specs)
        return console

    def _build_segment_header(self):
        h_specs = dict(title="Segment Distribution", height=20, width=500)
        header = TextBoxComponent(h_specs)
        return header

    def _build_console_header(self):
        h_specs = dict(title="Console", height=20, width=500)
        header = TextBoxComponent(h_specs)
        return header

    def _build_all_descr_header(self):
        h_specs = dict(title="Summed Scores to Each Descriptor", height=20,
                       width=500)
        header = TextBoxComponent(h_specs)
        return header

    def _build_enter_button(self):
        specs = dict()
        specs['text'] = 'Enter'
        specs['width'] = 100
        specs['height'] = 30
        button = ButtonComponent(specs, self.set_result_callback)
        return button

    def _init_result_CDS(self):
        data = dict(resno=["" for __ in range(90)],
                    pdb=["" for __ in range(90)],
                    cid=["" for __ in range(90)],
                    match_resno=["" for __ in range(90)],
                    match_vals=["" for __ in range(90)])
        source = ColumnDataSource(data)
        return source



    def _init_all_descr_CDS(self):
        data = dict(descr=list(self.descr_key_title_map.values()),
                    score=["0" for __ in range(5)])
        source = ColumnDataSource(data)
        return source

    def _init_summary_CDS(self):
        data = dict(resno=["" for __ in range(5)],
                    pdb=["" for __ in range(5)],
                    cid=["" for __ in range(5)],
                    match_resno=["" for __ in range(5)],
                    match_vals=["" for __ in range(5)])
        source = ColumnDataSource(data)
        return source


    def _init_descr_match_CDS(self):
        data = dict(resno=["" for __ in range(31)],
                    match_vals=["" for __ in range(31)])
        source = ColumnDataSource(data)
        return source

    def _build_summary_table_header(self):
        specs = dict(title="Consolidated Scores for Overall Match", height=7,
                     width=500)
        header = TextBoxComponent(specs)
        return header

    def _build_result_table_header(self):
        specs = dict(title="Scores at Individual Residue Positions",
                     height=7,
                     width=500)
        header = TextBoxComponent(specs)
        return header

    def _build_descr_match_table_header(self):
        specs = dict(title="Descriptor Match Scores",
                     height=7,
                     width=500)
        header = TextBoxComponent(specs)
        return header


    def _build_summary_table(self):
        columns = [
            TableColumn(field="pdb", title="PDB ID", width=40),
            TableColumn(field="cid", title="Chain ID", width=40),
            TableColumn(field="match_resno", title="Res_No", width=40),
            TableColumn(field="match_vals", title="Score", width=40)]
        data_table = DataTable(source=self.summary_CDS, columns=columns,
                               width=350, height=150, index_position=None,
                               fit_columns=True)
        return data_table

    def _build_all_descr_table(self):
        columns = [TableColumn(field="descr", title="Descriptor", width=40),
                   TableColumn(field="score", title="Score", width=40)]
        data_table = DataTable(source=self.all_descr_CDS, columns=columns,
                               width=350, height=150, index_position=None,
                               fit_columns=True)
        return data_table

    def _build_result_table(self):
        columns = [TableColumn(field="resno",
                               title="Input Residue No", width=80),
                   TableColumn(field="pdb", title="PDB ID", width=40),
                   TableColumn(field="cid", title="Chain ID", width=40),
                   TableColumn(field="match_resno", title="Res_No", width=40),
                   TableColumn(field="match_vals", title="Score",
                               width=40)]
        data_table = DataTable(source=self.result_CDS, columns=columns,
                               width=350,
                               height=280, index_position=None,
                               fit_columns=True)
        return data_table

    def _build_segment_plot(self):
        plot = IndividualFigure()
        return plot


    def _build_descr_match_table(self):
        columns = [TableColumn(field="resno", title="Res_No", width=40),
            TableColumn(field="match_vals", title="Score", width=40)]
        data_table = DataTable(source=self.descr_match_CDS, columns=columns,
                               width=350, height=150, index_position=None,
                               fit_columns=True)
        return data_table

    def _build_descr_key_title_map(self):
        mapping = OrderedDict()
        mapping['ef'] = 'EF-Hand'
        mapping['mg'] = 'MG-Binding DxDxDG'
        mapping['gxgxxg'] = 'GxGxxG'
        mapping['gxxgxg'] = 'GxxGxG'
        mapping['gxggxg'] = 'GxGGxG'
        mapping['single'] = 'single'
        return mapping

    def _build_descr_dropdown(self):
        descriptor_dropdown_options = dict()
        descriptor_dropdown_options['title'] = 'Descriptor'
        descriptor_dropdown_options['menu'] = [['ef', 'EF-Hand'],
                                               ['mg', 'MG-Binding DxDxDG'],
                                               ['gxgxxg', 'GxGxxG'],
                                               ['gxxgxg', 'GxxGxG'],
                                               ['gxggxg', 'GxGGxG'],
                                               ['single', 'single']]
        descriptor_dropdown_options['width'] = 123
        descriptor_dropdown_options['height'] = 50
        dropdown = DropDownComponent(descriptor_dropdown_options, self._set_descr_value)
        return dropdown

    def _build_seq_start_ti(self):
        text_input = TextInput(placeholder="12",
                               title="First Residue Number:", width=80)
        text_input.on_change("value", self._set_seq_start_value)
        return text_input

    def _build_pdb_ti(self):
        text_input = TextInput(placeholder="1abc", title="PDB Code:",
                               width=80)
        text_input.on_change("value", self._set_pdb_value)
        return text_input

    def _build_cid_ti(self):
        text_input = TextInput(placeholder="A", title="Chain ID:", width=80)
        text_input.on_change("value", self._set_cid_value)
        return text_input

    def _build_p_descr(self):
        specs = dict(title="Description<br><br>Enter the desired pdb code and "
                           "chain-id, select the descriptor you wish to match "
                           "against, and press Enter to obtain the match "
                           "results.",
                     height=50, width=500)
        descr = TextBoxComponent(specs)
        return descr

    def _build_p_header(self):
        specs = dict(title="Descriptor Matcher", height=30, width=500)
        header = TextBoxComponent(specs)
        return header

    def _build_figure(self):
        header_row = row(self.project_header.figure, height=30)
        description_row = row(self.project_description.figure, height=100)
        headers_col = column([header_row, description_row], height=120)
        summary_col = column([self.summary_table_header.figure,
                              self.summary_table], height=200)
        descr_match_col = column([self.descr_match_table_header.figure,
                                  self.descr_match_table], height=200)

        all_descr_col = column([self.all_descr_header.figure, self.all_descr_table])

        ind_result_col = column([self.result_table_header.figure,
                                 self.result_table])

        segment_col = column(
            [self.segment_header.figure, self.segment_plot.full_figure])

        result_col = column([summary_col, ind_result_col,
                             Div(height=20), descr_match_col,
                             all_descr_col, segment_col,
                             self.console_header.figure, self.console.figure])
        ti_col = column([self.pdb_ti, self.cid_ti, self.seq_start_ti,
                         self.descr_dropdown.figure,
                         self.enter_button.widget, self.msg_board.figure],
                        width=250, height=150)
        input_output_col = row([ti_col, result_col])
        figure = column([headers_col, input_output_col])
        return figure

# def show_plot():
#     ui = UI()
#     #curdoc().add_root(ui.figure)
#     return ui.figure



# p.image_rgba(image=[data], x=0, y=0, dw=1, dh=1, dilate=True)
# show(p)

ui = MatcherManager()
# ui.ui.ui_callback('gxgxxg', '1ZMC', "A", "178")
curdoc().add_root(ui.figure)
# show(ui.figure)

# ui = UI(empty_callback)
# curdoc().add_root(ui.figure)
# show(ui.figure)