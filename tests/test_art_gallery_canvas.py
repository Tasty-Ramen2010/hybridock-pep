"""Gallery pieces render at wildly different natural sizes (DNA 23x15,
peptide 44x19, docking/diffusion up to 29 wide with a ragged caption line).
Without normalizing, the TUI's surrounding "progress" panel border visibly
resized depending on which piece start_stream() happened to pick at random
for a given run. art._normalize_gallery_canvas() pads every frame of every
piece to one shared (width, height) canvas.
"""

from __future__ import annotations

from hybridock_pep.output import art


class TestGalleryCanvasIsUniform:
    def test_every_frame_of_every_piece_has_the_same_line_length(self):
        for idx in range(len(art.GALLERY)):
            _title, frames = art.gallery_piece(idx)
            widths = {len(ln) for f in frames for ln in f.split("\n")}
            assert len(widths) == 1, f"piece {idx} has ragged line widths: {widths}"

    def test_every_frame_of_every_piece_has_the_same_height(self):
        for idx in range(len(art.GALLERY)):
            _title, frames = art.gallery_piece(idx)
            heights = {len(f.split("\n")) for f in frames}
            assert len(heights) == 1, f"piece {idx} has varying heights: {heights}"

    def test_canvas_size_is_identical_across_every_piece(self):
        sizes = set()
        for idx in range(len(art.GALLERY)):
            _title, frames = art.gallery_piece(idx)
            f0 = frames[0]
            sizes.add((max(len(ln) for ln in f0.split("\n")), len(f0.split("\n"))))
        assert len(sizes) == 1, f"pieces render at different canvas sizes: {sizes}"

    def test_canvas_covers_the_largest_natural_piece(self):
        """The shared canvas must not crop anything — it should be at least
        as big as every piece's own natural (unpadded) extent."""
        _title, frames = art.gallery_piece(1)  # the peptide piece is the largest
        width, height = len(frames[0].split("\n")[0]), len(frames[0].split("\n"))
        assert width >= 44 and height >= 19


class TestPadFrame:
    def test_pads_a_smaller_frame_to_the_target_canvas(self):
        padded = art._pad_frame("ab\ncd", width=6, height=4)
        lines = padded.split("\n")
        assert len(lines) == 4
        assert all(len(ln) == 6 for ln in lines)

    def test_centers_content_within_the_canvas(self):
        padded = art._pad_frame("X", width=5, height=1)
        assert padded == "  X  "

    def test_equalizes_ragged_lines_within_one_frame_first(self):
        """A frame whose own lines differ in length (e.g. a short caption
        under a wider art block) must not distort centering — the shorter
        line is padded flush to the frame's own width before the whole block
        is centered in the shared canvas."""
        padded = art._pad_frame("abcde\nx", width=7, height=2)
        lines = padded.split("\n")
        assert len(lines[0]) == len(lines[1]) == 7

    def test_frame_already_at_target_size_is_unchanged_in_content(self):
        frame = "abc\ndef"
        padded = art._pad_frame(frame, width=3, height=2)
        assert padded == frame
