"""Test microtiming support in REMI tokenizer."""

from __future__ import annotations

from miditok import REMI, TokenizerConfig
from symusic import Note, Score, Tempo, TimeSignature, Track


def _get_mt_count(tokens) -> int:
    """Count MicroTiming tokens regardless of return type structure."""
    mt_count = 0
    if hasattr(tokens, "tokens"):
        # Single TokSequence
        for token in tokens.tokens:
            if isinstance(token, str) and token.startswith("MicroTiming"):
                mt_count += 1
    elif isinstance(tokens, list):
        for seq in tokens:
            if hasattr(seq, "tokens"):
                for token in seq.tokens:
                    if isinstance(token, str) and token.startswith("MicroTiming"):
                        mt_count += 1
            elif isinstance(seq, list):
                for token in seq:
                    if isinstance(token, str) and token.startswith("MicroTiming"):
                        mt_count += 1
    return mt_count


def test_remi_microtiming_roundtrip():
    """Test that microtiming round-trips correctly through encode-decode."""
    config = TokenizerConfig(
        pitch_range=(21, 109),
        beat_res={(0, 4): 12},
        num_velocities=8,
        special_tokens=["PAD", "BOS", "EOS", "MASK"],
        use_chords=False,
        use_rests=False,
        use_tempos=False,
        use_time_signatures=True,
        use_programs=False,
        use_microtiming=True,
        max_microtiming_shift=0.125,
        num_microtiming_bins=30,
    )
    tok = REMI(tokenizer_config=config)

    score = Score(tok.time_division)
    score.time_signatures = [TimeSignature(0, 4, 4)]
    score.tempos = [Tempo(0, 120)]

    track = Track(program=0, is_drum=False, name="Piano")
    track.notes.append(Note(0, 24, 60, 80))       # at pos 0, no microtiming
    track.notes.append(Note(2, 24, 62, 80))       # at pos 1, no microtiming
    track.notes.append(Note(5, 24, 64, 80))       # at pos 2+1 (microtiming=+1)
    track.notes.append(Note(8, 24, 65, 80))       # at pos 4, no microtiming
    score.tracks.append(track)

    tokens = tok.encode(score)
    score_decoded = tok.decode(tokens)

    for orig, decoded in zip(track.notes, score_decoded.tracks[0].notes):
        assert orig.start == decoded.start, (
            f"Start mismatch: {orig.start} != {decoded.start} "
            f"(pitch {orig.pitch})"
        )
        assert orig.pitch == decoded.pitch
        assert orig.velocity == decoded.velocity or abs(orig.velocity - decoded.velocity) <= 1

    mt_count = _get_mt_count(tokens)
    assert mt_count > 0, "No MicroTiming tokens were generated in roundtrip"


def test_remi_without_microtiming_still_works():
    """Test that REMI still works when microtiming is disabled (default)."""
    config = TokenizerConfig(
        pitch_range=(21, 109),
        beat_res={(0, 4): 12},
        num_velocities=8,
        special_tokens=["PAD", "BOS", "EOS", "MASK"],
        use_chords=False,
        use_rests=False,
        use_tempos=False,
        use_time_signatures=True,
        use_programs=False,
    )
    tok = REMI(tokenizer_config=config)

    score = Score(tok.time_division)
    score.time_signatures = [TimeSignature(0, 4, 4)]
    score.tempos = [Tempo(0, 120)]
    track = Track(program=0, is_drum=False, name="Piano")
    track.notes.append(Note(0, 24, 60, 80))
    track.notes.append(Note(2, 24, 62, 80))
    score.tracks.append(track)

    tokens = tok.encode(score)
    score_decoded = tok.decode(tokens)

    for orig, decoded in zip(track.notes, score_decoded.tracks[0].notes):
        assert orig.start == decoded.start
        assert orig.pitch == decoded.pitch


def test_microtiming_preserves_non_quantized_midi():
    """Test that microtiming tokens preserve subtle non-quantized timing."""
    config = TokenizerConfig(
        pitch_range=(21, 109),
        beat_res={(0, 4): 8},
        num_velocities=8,
        special_tokens=["PAD", "BOS", "EOS", "MASK"],
        use_chords=False,
        use_rests=False,
        use_tempos=False,
        use_time_signatures=True,
        use_programs=False,
        use_microtiming=True,
        max_microtiming_shift=0.125,
        num_microtiming_bins=30,
    )
    tok = REMI(tokenizer_config=config)

    score = Score(tok.time_division)
    score.time_signatures = [TimeSignature(0, 4, 4)]
    score.tempos = [Tempo(0, 120)]

    track = Track(program=0, is_drum=False, name="Piano")
    track.notes.append(Note(0, 16, 60, 64))
    track.notes.append(Note(3, 16, 62, 64))
    track.notes.append(Note(7, 16, 64, 64))
    track.notes.append(Note(9, 16, 65, 64))
    score.tracks.append(track)

    tokens = tok.encode(score)
    score_decoded = tok.decode(tokens)

    mt_count = _get_mt_count(tokens)
    assert mt_count > 0, "No MicroTiming tokens were generated"

    for orig, decoded in zip(track.notes, score_decoded.tracks[0].notes):
        diff = abs(orig.start - decoded.start)
        assert diff <= 1, (
            f"Start mismatch too large: {orig.start} != {decoded.start} "
            f"(diff={diff}, pitch={orig.pitch})"
        )


def test_microtiming_vocab_contains_tokens():
    """Test that the tokenizer vocabulary contains MicroTiming tokens."""
    config = TokenizerConfig(
        pitch_range=(21, 109),
        beat_res={(0, 4): 12},
        num_velocities=8,
        special_tokens=["PAD", "BOS", "EOS", "MASK"],
        use_chords=False,
        use_rests=False,
        use_tempos=False,
        use_time_signatures=True,
        use_programs=False,
        use_microtiming=True,
        max_microtiming_shift=0.125,
        num_microtiming_bins=30,
    )
    tok = REMI(tokenizer_config=config)

    mt_tokens = [t for t in tok.vocab if t.startswith("MicroTiming")]
    assert len(mt_tokens) > 0, "No MicroTiming tokens in vocabulary"
    for token in mt_tokens:
        assert "_" in token
        parts = token.split("_")
        assert parts[0] == "MicroTiming"
        int(parts[1])
