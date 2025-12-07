# Curates ARIA MIDI files for further processing
import sys
import pretty_midi

from lib.extractors import extract_chords, extract_high_melody, extract_key_estimates, extract_rythm
from lib.modifiers import remove_echo_and_split_tracks
from lib.interactive import play_midi_file_from_tracks
from lib.lahk import combine_tracks

def curate(filename: str, origin: str, action: str, interactive: bool) -> None:
    """
    Curates MIDI files by performing necessary preprocessing steps.

    Args:
        filename (str): The path to the ARIA MIDI file to be curated.
        origin (str): The origin of the MIDI file.
        interactive (bool): Whether to enable interactive playback during curation.
    """

    VALID_ORIGINS = ["aria", "lahk"]

    print(f"Curating {origin} MIDI file: {filename}, interactive mode: {interactive}, origin: {origin}")

    if origin not in VALID_ORIGINS:
        raise ValueError(f"Invalid origin '{origin}'. Valid origins are: {VALID_ORIGINS}")
    
    VALID_ACTIONS = {
        "lahk": [
            "curate",
            "ariafy",
        ],
        "aria": [
            "curate",
        ]
    }

    if action not in VALID_ACTIONS[origin]:
        raise ValueError(f"Invalid action '{action}' for origin '{origin}'. Valid actions are: {VALID_ACTIONS[origin]}")
    
    # TODO implement curate action for lahk origin
    # as by default it just ariafies it

    parsed = pretty_midi.PrettyMIDI(filename)
    track = None
    drum_track = None
    left_hand_track = None
    right_hand_track = None
    combined_track_with_echo = None
    qsize = None
    most_common_note_duration = None
    microchord_artifact_size = 0.05  # 50 ms
    if origin == "aria":
        track = parsed.instruments[0]
        left_hand_track, right_hand_track, combined_track_with_echo, qsize, most_common_note_duration = remove_echo_and_split_tracks(track)
        microchord_artifact_size = qsize * 3.0
    elif origin == "lahk":
        left_hand_track, right_hand_track, combined_track_with_echo, drum_track, most_common_note_duration = combine_tracks(parsed.instruments)

    key_estimates_grouped = extract_key_estimates(combined_track_with_echo) 
    for estimate in key_estimates_grouped:
        right_hand_track = estimate.get_track_without_notes_to_drop(right_hand_track)
        left_hand_track = estimate.get_track_without_notes_to_drop(left_hand_track)
        combined_track_with_echo = estimate.get_track_without_notes_to_drop(combined_track_with_echo)

    (
        chord_track,
        lyrics,
        combined_chords,
    ) = extract_chords(combined_track_with_echo, key_estimates_grouped, microchord_artifact_size, most_common_note_duration)

    highest_melody_track = extract_high_melody(right_hand_track)  # Placeholder for actual highest melody extraction
    #rythm_track = extract_rythm(right_hand_track, left_hand_track, qsize)  # Placeholder for actual rhythm track extraction

    if interactive:
        play_midi_file_from_tracks(
            [left_hand_track, right_hand_track, highest_melody_track, combined_track_with_echo, chord_track, drum_track],
            [ks.get_signature() for ks in key_estimates_grouped],
            lyrics,
            parsed if origin == "lahk" else None,
        )
    
if __name__ == "__main__":
    file_to_curate = sys.argv[3]
    origin = sys.argv[1]
    action = sys.argv[2]
    curate(file_to_curate, origin, action, True)