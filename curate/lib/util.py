import pretty_midi

def display_set_of_notes(notes_set):
    """Utility function to display a set of notes in a readable format."""
    note_names = [pretty_midi.note_number_to_name(note) for note in notes_set]
    return "{" + ", ".join(note_names) + "}"