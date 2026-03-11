import pytest
from core.models import CutPoint
from ffmpeg.filter_graph import FilterGraphBuilder


def make_cuts():
    return [
        CutPoint(2.0, 5.0, 0, "hard_cut", 0.0),
        CutPoint(12.0, 16.0, 0, "crossfade", 0.3),
        CutPoint(22.0, 25.0, 0, "wipe_left", 0.2),
    ]


class TestFilterGraphBuilder:
    def test_group_by_transitions(self):
        cuts = make_cuts()
        builder = FilterGraphBuilder()
        groups = builder.group_cuts(cuts)
        # First cut has hard_cut transition -> boundary
        # Cuts 1-2 have non-hard transitions -> one group
        assert len(groups) >= 1

    def test_build_xfade_graph_single_source(self):
        cuts = [
            CutPoint(2.0, 5.0, 0, "crossfade", 0.3),
            CutPoint(12.0, 16.0, 0, "crossfade", 0.3),
            CutPoint(22.0, 25.0, 0, "crossfade", 0.3),
        ]
        builder = FilterGraphBuilder()
        graph = builder.build_xfade_graph(cuts, source_mapping={0: 0})
        assert "xfade" in graph
        assert "trim" in graph
        assert "setpts" in graph

    def test_offset_calculation(self):
        builder = FilterGraphBuilder()
        # Segment 0: 3s, transition 0: 0.3s => offset_0 = 2.7
        # Segment 1: 4s, transition 1: 0.2s => offset_1 = 2.7 + 4.0 - 0.2 = 6.5
        cuts = [
            CutPoint(2.0, 5.0, 0, "crossfade", 0.3),
            CutPoint(12.0, 16.0, 0, "wipe_left", 0.2),
            CutPoint(22.0, 25.0, 0, "hard_cut", 0.0),  # last, no xfade after
        ]
        offsets = builder.calculate_offsets(cuts)
        assert abs(offsets[0] - 2.7) < 0.001
        assert abs(offsets[1] - 6.5) < 0.001

    def test_transition_mapping(self):
        builder = FilterGraphBuilder()
        assert builder.xfade_name("crossfade") == "fade"
        assert builder.xfade_name("crossfade_slow") == "fade"
        assert builder.xfade_name("fade_black") == "fadeblack"
        assert builder.xfade_name("wipe_left") == "wipeleft"
        assert builder.xfade_name("wipe_right") == "wiperight"

    def test_batch_splitting(self):
        # 25 cuts with non-hard transitions -> should split at 20
        cuts = [
            CutPoint(float(i), float(i + 1), 0, "crossfade", 0.3)
            for i in range(25)
        ]
        builder = FilterGraphBuilder()
        batches = builder.split_batches(cuts, max_per_batch=20)
        assert len(batches) == 2
        assert len(batches[0]) == 20
        assert len(batches[1]) == 5

    def test_multi_source_mapping(self):
        cuts = [
            CutPoint(5.0, 8.0, 0, "crossfade", 0.3),
            CutPoint(2.0, 6.0, 2, "crossfade", 0.3),
            CutPoint(15.0, 18.0, 0, "hard_cut", 0.0),
        ]
        builder = FilterGraphBuilder()
        mapping = builder.build_source_mapping(cuts)
        # Should map original indices 0, 2 to batch-local 0, 1
        assert 0 in mapping
        assert 2 in mapping
        assert mapping[0] == 0
        assert mapping[2] == 1
