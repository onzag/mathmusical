import pretty_midi
import code
import math

MAX_NOTE_PITCH_GAP_SAMEHAND = 11  # in MIDI pitch numbers
NOTE_SAME_TIME_THRESHOLD = 0.05  # in seconds
NOTE_FORCE_PAD_THRESHOLD = 0.1  # in seconds

def note_belongs_to_same_hand(
    last_note: pretty_midi.Note,
    following_note: pretty_midi.Note,
) -> tuple[int, bool]:
    """
    Determines if two notes belong to the same hand based on their pitch.

    Args:
        note1 (pretty_midi.Note): The first MIDI note.
        note2 (pretty_midi.Note): The second MIDI note.

    Returns:
        bool: True if both notes belong to the same hand, False otherwise.
    """
    if last_note is None:
        return (0, True)

    # Checks whether two notes belong to the same hand based on pitch
    diff = abs(last_note.pitch - following_note.pitch)
    belongs = (diff <= MAX_NOTE_PITCH_GAP_SAMEHAND)
    return (int(diff), belongs)

def calculate_average_pitch(track: pretty_midi.Instrument | list[pretty_midi.Note]) -> float:
    """
    Calculates the average pitch of all notes in a given MIDI track.

    Args:
        track (pretty_midi.Instrument): The MIDI track to analyze.
    Returns:
        float: The average pitch of the notes in the track. Returns 0.0 if
                the track has no notes.
    if len(track.notes) == 0:
    """

    if (isinstance(track, pretty_midi.Instrument) and len(track.notes) == 0) or (isinstance(track, list) and len(track) == 0):
        return 0.0
    total_pitch = sum(note.pitch for note in track.notes) if isinstance(track, pretty_midi.Instrument) else sum(note.pitch for note in track)
    average_pitch = total_pitch / (len(track.notes) if isinstance(track, pretty_midi.Instrument) else len(track))
    return average_pitch

def note_is_starting_simultaneously(
    note1: pretty_midi.Note,
    note2: pretty_midi.Note,
) -> bool:
    """
    Determines if two notes are playing simultaneously based on their start time

    Args:
        note1 (pretty_midi.Note): The first MIDI note.
        note2 (pretty_midi.Note): The second MIDI note.

    Returns:
        bool: True if both notes are playing simultaneously, False otherwise.
    """
    return math.fabs(note1.start - note2.start) <= NOTE_SAME_TIME_THRESHOLD

def note_is_ending_simultaneously(
    note1: pretty_midi.Note,
    note2: pretty_midi.Note,
) -> bool:
    """
    Determines if two notes are ending simultaneously based on their end times.
    """
    return math.fabs(note1.end - note2.end) <= NOTE_SAME_TIME_THRESHOLD

def note_ends_while_other_starts(
    prev_note: pretty_midi.Note,
    next_note: pretty_midi.Note,
) -> bool:
    """
    Determines if one note ends while another starts within a defined threshold.

    Args:
        prev_note (pretty_midi.Note): The first MIDI note.
        next_note (pretty_midi.Note): The second MIDI note.

    Returns:
        bool: True if one note ends while the other starts within the threshold, False otherwise.
    """
    return math.fabs(next_note.start - prev_note.end) <= NOTE_SAME_TIME_THRESHOLD

def note_ends_while_other_starts_non_exact(
    prev_note: pretty_midi.Note,
    next_note: pretty_midi.Note,
) -> bool:
    """
    Determines if one note ends while another starts within a more lenient threshold.

    Args:
        prev_note (pretty_midi.Note): The first MIDI note.
        next_note (pretty_midi.Note): The second MIDI note.

    Returns:
        bool: True if one note ends while the other starts within the lenient threshold, False otherwise.
    """
    diff = math.fabs(next_note.start - prev_note.end)
    if diff == 0:
        return False
    return diff <= NOTE_SAME_TIME_THRESHOLD

def remove_echo_and_split_tracks(track: pretty_midi.Instrument) -> tuple[pretty_midi.Instrument, pretty_midi.Instrument]:
    """
    Removes echo effects from a given MIDI track by eliminating notes that are
    very close in time to preceding notes.

    Args:
        track (pretty_midi.Instrument): The MIDI track to process.

    Because this is optimized for the ARIA MIDI files, it assumes that echoes are
    defined as piano notes, therefore only accepts 2 notes being played at the same time and
    will split them accordingly.
    """
    # first, lets make a copy of the track
    for i, note in enumerate(track.notes):
        previous_notes = track.notes[:i][-10:]  # get the last ten previous notes, to compare against, we have 10 fingers after all
        for previous_note in previous_notes:
            if note_is_starting_simultaneously(previous_note, note):
                # If two notes are playing simultaneously, we keep both, we will however take the min_start and max_end
                min_start = min(previous_note.start, note.start)
                previous_note.start = min_start
                note.start = min_start
            if note_is_ending_simultaneously(previous_note, note):
                # If two notes are ending simultaneously, we keep both, we will however take the max_end
                max_end = max(previous_note.end, note.end)
                previous_note.end = max_end
                note.end = max_end

    track.notes.sort(key=lambda n: n.start)

    for i, note in enumerate(track.notes):
        previous_notes = track.notes[:i][-10:]  # get the last ten previous notes, to compare against, we have 10 fingers after all
        for previous_note in previous_notes:
            if note_ends_while_other_starts_non_exact(previous_note, note):
                # If one note ends while the other starts, we adjust the end of the previous note to the start of the next note
                new_end = note.start
                if previous_note.start <= new_end:
                    # weird situation, we will assume they are both supposed to be played at the same time
                    # we will crop the previous note to half the duration just in case
                    new_end = max(note.end / 2, previous_note.end)

                previous_note.end = new_end

    # remove potentially invalid notes that may have appeared due to cropping
    # or that are too short
    track.notes = [note for note in track.notes if note.end > note.start and (note.end - note.start) >= (NOTE_SAME_TIME_THRESHOLD / 2)]

    track_A = pretty_midi.Instrument(program=track.program, is_drum=False, name="track A")
    track_B = pretty_midi.Instrument(program=track.program, is_drum=False, name="track B")

    #code.interact(local=locals())

    # let's start looping through notes in order as they appear
    for note in track.notes:
        last_note_A = track_A.notes[-1] if len(track_A.notes) > 0 else None
        last_note_B = track_B.notes[-1] if len(track_B.notes) > 0 else None
        
        belongs_to_A_diff, belongs_to_A = note_belongs_to_same_hand(last_note_A, note)
        belongs_to_B_diff, belongs_to_B = note_belongs_to_same_hand(last_note_B, note)

        if belongs_to_A and (not belongs_to_B or belongs_to_A_diff <= belongs_to_B_diff):
            track_A.notes.append(note)
        elif belongs_to_B:
            track_B.notes.append(note)
        else:
            # If it doesn't belong to either hand, we are going to check against the average pitch
            # of the last 10 notes in each track
            last_10_notes_A = track_A.notes[-10:] if len(track_A.notes) >= 10 else track_A.notes
            last_10_notes_B = track_B.notes[-10:] if len(track_B.notes) >= 10 else track_B.notes
            avg_pitch_last_10_A = calculate_average_pitch(last_10_notes_A)
            avg_pitch_last_10_B = calculate_average_pitch(last_10_notes_B)
            # append to the closest average pitch to the current note
            if abs(note.pitch - avg_pitch_last_10_A) <= abs(note.pitch - avg_pitch_last_10_B):
                track_A.notes.append(note)
            else:
                track_B.notes.append(note)

    track_A.notes.sort(key=lambda n: n.start)
    track_B.notes.sort(key=lambda n: n.start)

    # now we will calculate the average pitches to decide which track is left and which is right
    avg_pitch_A = calculate_average_pitch(track_A)
    avg_pitch_B = calculate_average_pitch(track_B)
    if avg_pitch_A <= avg_pitch_B:
        left_hand_track = track_A
        right_hand_track = track_B
    else:
        left_hand_track = track_B
        right_hand_track = track_A

    # rename tracks accordingly
    left_hand_track.name = "Left Hand Track"
    right_hand_track.name = "Right Hand Track"

    for track in [left_hand_track, right_hand_track]:
        # we want now to crop echoing notes that are very close in time to preceding notes in the same track, aka they overlap in time
        for i, note in enumerate(track.notes):
            previous_notes = track.notes[:i][-10:]  # get the last ten previous notes, to compare against, we have 10 fingers after all
            for prev_note in previous_notes:
                # we are going to check whether it overlaps in time with the prev, as in the end time is less than the start time of the next note plus a threshold
                if prev_note.end > note.start:
                    if not note_is_starting_simultaneously(prev_note, note):
                        new_end = note.start
                        prev_note.end = new_end  # crop the previous note to end when the next note starts
                    # simultaneous notes
                    else:
                        # the note that ends later will be cropped to the end time of the note that ends earlier
                        # basically this ensure that if the previous note is too long, it will be cropped to the end time of the next note
                        # but if it is too short within a threshold, it will be extended to the end time of the next note
                        if prev_note.end + NOTE_FORCE_PAD_THRESHOLD >= note.end:
                            prev_note.end = note.end
                        else:
                            # otherwise we let it be as it is
                            pass

    # let's now find the note durations that are most common for the NOTE_SAME_TIME_THRESHOLD for the same start and end times
    #notes_groups = []
    #for note in track.notes:
    #    found = False
        # loop through existing groups to see if the note fits in any in the window given
    #    for note_group in notes_groups:
    #        for note_in_group in note_group:
    #            if note_is_starting_simultaneously(note_in_group, note) and note_is_ending_simultaneously(note_in_group, note):
    #                note_group.append(note)
    #                found = True
    #                break
    #        if found:
    #            break
        # not found anywhere
    #    if not found:
    #        note_group = [note]
    #        notes_groups.append(note_group)
                    
    return (left_hand_track, right_hand_track)