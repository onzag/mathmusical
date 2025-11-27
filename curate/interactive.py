import pretty_midi
import os
import platform
import subprocess
import tempfile

def open_with_default_application(file_path: str) -> None:
    """
    Opens a file with the system's default application for that file type.

    Args:
        file_path (str): The path to the file to be opened.
    """

    if platform.system() == "Windows":
        os.startfile(file_path)
    elif platform.system() == "Darwin":  # macOS
        subprocess.call(["open", file_path])
    else:  # Linux and other Unix-like systems
        subprocess.call(["xdg-open", file_path])

def play_midi_file_from_tracks(tracks: list[pretty_midi.Instrument]) -> None:
    """
    Plays a MIDI file constructed from the given tracks.
    Args:
        tracks (list[pretty_midi.Instrument]): List of MIDI instrument tracks.
    """
    midi_data = pretty_midi.PrettyMIDI()
    list_order = [track for track in tracks]
    for track in list_order.__reversed__():
        if track is not None:
            midi_data.instruments.append(track)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mid") as temp_midi_file:
        midi_data.write(temp_midi_file.name)
        temp_midi_file_path = temp_midi_file.name
    open_with_default_application(temp_midi_file_path)