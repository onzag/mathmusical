# Curates ARIA MIDI files for further processing
import sys
import pretty_midi

from lib.extractors import extract_high_melody, extract_key_estimates
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

    left_hand_track, right_hand_track, qsize = remove_echo_and_split_tracks(track)

    key_estimates_grouped = extract_key_estimates(right_hand_track, left_hand_track)  # Placeholder for actual key estimation extraction
    for estimate in key_estimates_grouped:
        right_hand_track = estimate.get_track_without_notes_to_drop(right_hand_track)
        left_hand_track = estimate.get_track_without_notes_to_drop(left_hand_track)
        print(estimate)

    highest_melody_track = extract_high_melody(right_hand_track)  # Placeholder for actual highest melody extraction

    if interactive:
        play_midi_file_from_tracks([left_hand_track, right_hand_track, highest_melody_track], [ks.get_signature() for ks in key_estimates_grouped])
    
if __name__ == "__main__":
    file_to_curate = sys.argv[1]
    curate_aria_midi(file_to_curate, True)