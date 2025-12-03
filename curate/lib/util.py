import pretty_midi
from music21.chord import Chord

def display_set_of_notes(notes_set):
    """Utility function to display a set of notes in a readable format."""
    note_names = [pretty_midi.note_number_to_name(note) for note in notes_set]
    return "{" + ", ".join(note_names) + "}"

def get_chord_name(notes_set):
    """Utility function to get a chord name from a set of notes."""
    # This is a placeholder implementation. A real implementation would analyze the notes.
    chord = Chord(list(notes_set))
    chordName = chord.pitchedCommonName
    chordName = chordName.replace(" triad", "").replace(" chord", "").replace("-major", "").replace("-minor", "m")
    chordName = chordName.replace("-augmented", "aug").replace("-diminished", "dim").replace(" seventh", "7").replace(" fifth", "5")
    chordName = chordName.replace("Major", "maj").replace("Minor", "min")
    chordName = chordName.replace(" Sixth", "6").replace(" above ", "^").replace(" suspended ", "sus").replace("-suspended ", "sus")
    return chordName