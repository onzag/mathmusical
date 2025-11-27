# Curates ARIA MIDI files for further processing
import sys
import pretty_midi

from modifiers import remove_echo_and_split_tracks
from interactive import play_midi_file_from_tracks

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

    left_hand_track, right_hand_track = remove_echo_and_split_tracks(track)

    if interactive:
        play_midi_file_from_tracks([left_hand_track, right_hand_track])
    
if __name__ == "__main__":
    file_to_curate = sys.argv[1]
    curate_aria_midi(file_to_curate, True)