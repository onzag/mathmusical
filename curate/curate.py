# Curates ARIA MIDI files for further processing
import sys
import pretty_midi

from lib.extractors import extract_chords, extract_high_melody, extract_key_estimates, extract_rythm
from lib.modifiers import remove_echo_and_split_tracks
from lib.interactive import play_midi_file_from_tracks

def curate_aria_midi(filename: str, interactive: bool) -> None:
    """
    Curates ARIA MIDI files by performing necessary preprocessing steps.

    Args:
        filename (str): The path to the ARIA MIDI file to be curated.
    """
    # Placeholder for actual curation logic
    print(f"Curating ARIA MIDI file: {filename}, interactive mode: {interactive}")

    parsed = pretty_midi.PrettyMIDI(filename)
    track = parsed.instruments[0]
    print(f"Track number: {0}, Number of notes: {len(track.notes)}")

    left_hand_track, right_hand_track, combined_track_with_echo, qsize, most_common_note_duration = remove_echo_and_split_tracks(track)

    key_estimates_grouped = extract_key_estimates(combined_track_with_echo)  # Placeholder for actual key estimation extraction
    for estimate in key_estimates_grouped:
        right_hand_track = estimate.get_track_without_notes_to_drop(right_hand_track)
        left_hand_track = estimate.get_track_without_notes_to_drop(left_hand_track)
        combined_track_with_echo = estimate.get_track_without_notes_to_drop(combined_track_with_echo)

    (
        chord_track,
        lyrics,
        combined_chords,
        structure_with_best_match,
        current_best_matches_score,
    ) = extract_chords(combined_track_with_echo, key_estimates_grouped, qsize, most_common_note_duration)  # Placeholder for actual chord extraction

    highest_melody_track = extract_high_melody(right_hand_track)  # Placeholder for actual highest melody extraction
    #rythm_track = extract_rythm(right_hand_track, left_hand_track, qsize)  # Placeholder for actual rhythm track extraction

    if interactive:
        play_midi_file_from_tracks(
            [left_hand_track, right_hand_track, highest_melody_track, combined_track_with_echo, chord_track],
            [ks.get_signature() for ks in key_estimates_grouped],
            lyrics,
        )
    
if __name__ == "__main__":
    file_to_curate = sys.argv[1]
    curate_aria_midi(file_to_curate, True)