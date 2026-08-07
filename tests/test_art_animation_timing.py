"""frame_index_for_elapsed() — the pure-function clock behind the TUI's
continuously-looping Stage-1 gallery animation (tui.py's refresh_progress).

Before this, the loop used a plain `int(elapsed / frame_delay) % len(frames)`
modulo, so every frame (including the first and last) got exactly the same
on-screen time and the loop snapped straight from the last frame back to the
first with no beat at either end — reported as the animation feeling "too
fast" and inconsistent. frame_index_for_elapsed adds an explicit extra hold
on frame 0 and frame -1 of each cycle.
"""

from __future__ import annotations

from hybridock_pep.output import art


class TestFrameIndexForElapsed:
    def test_single_frame_is_always_index_zero(self):
        for elapsed in (0.0, 1.0, 100.0):
            assert art.frame_index_for_elapsed(1, elapsed) == 0

    def test_zero_frames_does_not_raise(self):
        """Defensive: gallery_piece() never actually returns an empty frame
        tuple, but the function must not crash if it somehow did."""
        assert art.frame_index_for_elapsed(0, 5.0) == 0

    def test_frame_zero_at_time_zero(self):
        assert art.frame_index_for_elapsed(4, 0.0, frame_delay=0.32) == 0

    def test_first_frame_holds_for_delay_plus_extra(self):
        """With hold_first=1.0 and frame_delay=0.32, frame 0 must still be
        showing at t=1.0s (a plain modulo would have already advanced to
        frame 3 by then: 1.0 / 0.32 = 3.1)."""
        n = 4
        fi = art.frame_index_for_elapsed(n, 1.0, frame_delay=0.32, hold_first=1.0, hold_last=1.0)
        assert fi == 0

    def test_advances_past_first_frame_once_its_hold_elapses(self):
        n = 4
        just_after_hold = 0.32 + 1.0 + 0.01
        fi = art.frame_index_for_elapsed(n, just_after_hold, frame_delay=0.32,
                                         hold_first=1.0, hold_last=1.0)
        assert fi == 1

    def test_last_frame_holds_before_wrapping_to_first(self):
        """Total cycle = 2 non-edge frames * 0.32 + (0.32+1.0) first + (0.32+1.0) last.
        Just before the cycle ends, we must still be on the last frame index."""
        n = 4
        frame_delay, hold = 0.32, 1.0
        total = (n - 2) * frame_delay + (frame_delay + hold) * 2
        just_before_wrap = total - 0.01
        fi = art.frame_index_for_elapsed(n, just_before_wrap, frame_delay=frame_delay,
                                         hold_first=hold, hold_last=hold)
        assert fi == n - 1

    def test_wraps_back_to_first_frame_after_a_full_cycle(self):
        n = 4
        frame_delay, hold = 0.32, 1.0
        total = (n - 2) * frame_delay + (frame_delay + hold) * 2
        assert art.frame_index_for_elapsed(n, total + 0.001, frame_delay, hold, hold) == 0
        assert art.frame_index_for_elapsed(n, total * 3 + 0.001, frame_delay, hold, hold) == 0

    def test_middle_frames_get_only_the_plain_delay(self):
        """A middle frame (not first, not last) must not receive either hold —
        it should occupy exactly `frame_delay` seconds of the cycle."""
        n = 5
        frame_delay, hold = 0.32, 1.0
        t_frame0_end = frame_delay + hold
        t_frame1_end = t_frame0_end + frame_delay
        fi_at_start = art.frame_index_for_elapsed(n, t_frame0_end + 0.001, frame_delay, hold, hold)
        fi_at_end = art.frame_index_for_elapsed(n, t_frame1_end - 0.001, frame_delay, hold, hold)
        assert fi_at_start == fi_at_end == 1

    def test_default_holds_match_module_constants(self):
        """The TUI relies on the module-level defaults matching
        HOLD_FIRST_FRAME_S / HOLD_LAST_FRAME_S so it does not have to pass
        them explicitly on every call."""
        n = 3
        t = art.HOLD_FIRST_FRAME_S + 0.05
        assert art.frame_index_for_elapsed(n, t) == 0
        t2 = art.HOLD_FIRST_FRAME_S + 0.32 + 0.05
        assert art.frame_index_for_elapsed(n, t2) == 1

    def test_result_always_in_range(self):
        n = 6
        for elapsed in (0.0, 0.1, 0.5, 1.0, 3.0, 10.0, 1000.0):
            fi = art.frame_index_for_elapsed(n, elapsed)
            assert 0 <= fi < n

    def test_is_deterministic(self):
        assert (art.frame_index_for_elapsed(6, 4.2)
                == art.frame_index_for_elapsed(6, 4.2))


class TestGalleryPieceFramesWorkWithTheTimingHelper:
    """Every real gallery piece must be a valid input, not just synthetic ns."""

    def test_every_gallery_piece_produces_a_valid_index_across_a_cycle(self):
        for idx in range(len(art.GALLERY)):
            _title, frames = art.gallery_piece(idx)
            n = len(frames)
            for elapsed in (0.0, 0.05, 0.5, 1.5, 3.0, 10.0):
                fi = art.frame_index_for_elapsed(n, elapsed)
                assert 0 <= fi < n
