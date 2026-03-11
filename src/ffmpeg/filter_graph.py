import logging
from core.models import CutPoint

logger = logging.getLogger("autoeditor.ffmpeg")

XFADE_MAP = {
    "crossfade": "fade",
    "crossfade_slow": "fade",
    "fade_black": "fadeblack",
    "wipe_left": "wipeleft",
    "wipe_right": "wiperight",
}


class FilterGraphBuilder:
    def xfade_name(self, transition_type: str) -> str:
        return XFADE_MAP.get(transition_type, "fade")

    def group_cuts(self, cuts: list[CutPoint]) -> list[list[CutPoint]]:
        """Group cuts by transition boundaries.
        Convention: CutPoint.transition_type = transition FROM this segment to the next.
        A hard_cut on cuts[i] means a hard cut AFTER segment i, so segment i+1 starts a new group."""
        if not cuts:
            return []
        groups = []
        current_group = [cuts[0]]
        for i in range(1, len(cuts)):
            # Check if the PREVIOUS segment has a hard_cut (transition after it)
            if cuts[i - 1].transition_type == "hard_cut":
                groups.append(current_group)
                current_group = [cuts[i]]
            else:
                current_group.append(cuts[i])
        groups.append(current_group)
        return groups

    def split_batches(self, cuts: list[CutPoint], max_per_batch: int = 20) -> list[list[CutPoint]]:
        if len(cuts) <= max_per_batch:
            return [cuts]
        batches = []
        for i in range(0, len(cuts), max_per_batch):
            batches.append(cuts[i:i + max_per_batch])
        return batches

    def build_source_mapping(self, cuts: list[CutPoint]) -> dict[int, int]:
        unique_sources = sorted(set(c.source_index for c in cuts))
        return {orig: local for local, orig in enumerate(unique_sources)}

    def calculate_offsets(self, cuts: list[CutPoint]) -> list[float]:
        if len(cuts) < 2:
            return []
        offsets = []
        seg0_dur = cuts[0].end - cuts[0].start
        offset = seg0_dur - cuts[0].transition_duration
        offsets.append(offset)
        for i in range(1, len(cuts) - 1):
            seg_dur = cuts[i].end - cuts[i].start
            offset = offset + seg_dur - cuts[i].transition_duration
            offsets.append(offset)
        return offsets

    def build_xfade_graph(self, cuts: list[CutPoint], source_mapping: dict[int, int]) -> str:
        if len(cuts) < 2:
            c = cuts[0]
            local_idx = source_mapping[c.source_index]
            return f"[{local_idx}:v]trim=start={c.start}:end={c.end},setpts=PTS-STARTPTS[vout]"

        lines = []
        # Trim each segment
        for i, c in enumerate(cuts):
            local_idx = source_mapping[c.source_index]
            lines.append(f"[{local_idx}:v]trim=start={c.start}:end={c.end},setpts=PTS-STARTPTS[v{i}]")

        # Chain xfade transitions
        offsets = self.calculate_offsets(cuts)
        prev_label = "[v0]"
        for i in range(len(cuts) - 1):
            xfade = self.xfade_name(cuts[i].transition_type)
            dur = cuts[i].transition_duration
            offset = offsets[i]
            if i == len(cuts) - 2:
                out_label = "[vout]"
            else:
                out_label = f"[vt{i + 1}]"
            lines.append(
                f"{prev_label}[v{i + 1}]xfade=transition={xfade}:duration={dur}:offset={offset}{out_label}"
            )
            prev_label = out_label

        return ";\n".join(lines)

    def build_audio_graph(
        self, cuts: list[CutPoint], source_mapping: dict[int, int], use_separate_audio: bool,
    ) -> str | None:
        if use_separate_audio:
            return None  # Audio is laid down as-is from the separate file

        if len(cuts) < 2:
            c = cuts[0]
            local_idx = source_mapping[c.source_index]
            return f"[{local_idx}:a]atrim=start={c.start}:end={c.end},asetpts=PTS-STARTPTS[aout]"

        lines = []
        for i, c in enumerate(cuts):
            local_idx = source_mapping[c.source_index]
            # Compute overlap extensions per spec:
            # Outgoing: extend atrim_end by outgoing transition duration
            # Incoming: start atrim earlier by incoming transition duration
            out_t = c.transition_duration if i < len(cuts) - 1 else 0.0
            in_t = cuts[i - 1].transition_duration if i > 0 else 0.0
            # For hard cuts, use 30ms overlap
            if out_t == 0.0 and i < len(cuts) - 1:
                out_t = 0.03
            if in_t == 0.0 and i > 0:
                in_t = 0.03
            atrim_start = max(0, c.start - in_t)
            atrim_end = c.end + out_t
            lines.append(
                f"[{local_idx}:a]atrim=start={atrim_start}:end={atrim_end},asetpts=PTS-STARTPTS[a{i}]"
            )

        # Chain acrossfade
        prev_label = "[a0]"
        for i in range(len(cuts) - 1):
            dur = cuts[i].transition_duration
            if dur == 0.0:
                dur = 0.03  # 30ms for hard cuts
            if i == len(cuts) - 2:
                out_label = "[aout]"
            else:
                out_label = f"[at{i + 1}]"
            lines.append(f"{prev_label}[a{i + 1}]acrossfade=d={dur}{out_label}")
            prev_label = out_label

        return ";\n".join(lines)

    def build_concat_file(self, file_paths: list[str]) -> str:
        lines = []
        for path in file_paths:
            escaped = path.replace("'", "'\\''")
            lines.append(f"file '{escaped}'")
        return "\n".join(lines)
