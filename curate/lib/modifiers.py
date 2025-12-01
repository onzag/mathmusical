import pretty_midi
import code
import math

MAX_NOTE_PITCH_GAP_SAMEHAND = 11  # in MIDI pitch numbers
MAX_NOTE_VELOCITY_GAP_SAMEHAND = 30  # in MIDI velocity numbers
NOTE_SAME_TIME_THRESHOLD = 0.05  # in seconds
NOTE_FORCE_PAD_THRESHOLD = 0.1  # in seconds
NOTE_CLOSE_IN_TIME_THRESHOLD = 0.2  # in seconds

MELODY_WEIGHT_PITCH_SIMILARITY = 0.3
MELODY_WEIGHT_VELOCITY_SIMILARITY = 0.3
MELODY_WEIGHT_HIGHEST_NOTE = 0.5
MELODY_WEIGHT_ONLY_NOTE = 0.5
MELODY_WEIGHT_TIME_SIMILARITY = 0.5

MELODY_SAMETIME_NOTE_AMOUNT_PENALTY_STARTS = 2
MELODY_SAMETIME_NOTE_AMOUNT_PENALTY = 2.0

MELODY_WEIGHT_NON_HIGHEST_NOTE_PENALTY = 0.5
MELODY_WEIGHT_LOWEST_NOTE_PENALTY = 1.2
MELODY_WEIGHT_LARGE_GAP_DOWN_PENALTY = 1.2

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

def calculate_average_velocity(track: pretty_midi.Instrument) -> float:
    """
    Calculates the average velocity of all notes in a given MIDI track.

    Args:
        track (pretty_midi.Instrument): The MIDI track to analyze.
    Returns:
        float: The average velocity of the notes in the track. Returns 0.0 if
                the track has no notes.
    """
    if len(track.notes) == 0:
        return 0.0
    total_velocity = sum(note.velocity for note in track.notes)
    average_velocity = total_velocity / len(track.notes)
    return average_velocity

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

def note_has_same_duration(
    note1: pretty_midi.Note,
    note2: pretty_midi.Note,
) -> bool:
    """
    Determines if two notes have the same duration based on their start and end times.
    duration1 = note1.end - note1.start
    duration2 = note2.end - note2.start
    """
    duration1 = note1.end - note1.start
    duration2 = note2.end - note2.start
    return math.fabs(duration1 - duration2) <= NOTE_SAME_TIME_THRESHOLD


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

def remove_echo(track: pretty_midi.Instrument):
    """
    Removes echo effects from a given MIDI track by eliminating notes that are
    very close in time to preceding notes.
    """

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

def move_melody_to_right_hand(
    left_hand_track: pretty_midi.Instrument,
    right_hand_track: pretty_midi.Instrument,
) -> tuple[pretty_midi.Instrument, pretty_midi.Instrument]:
    pass
    """
    Moves melody notes to the right hand track based on their pitch.

    Args:
        left_hand_track (pretty_midi.Instrument): The left hand MIDI track.
        right_hand_track (pretty_midi.Instrument): The right hand MIDI track.
        avg_pitch_right_hand (float): The average pitch of the right hand track.
    """
    avg_velocity_right_hand = calculate_average_velocity(right_hand_track)
    avg_pitch_right_hand = calculate_average_pitch(right_hand_track)
    avg_velocity_left_hand = calculate_average_velocity(left_hand_track)
    avg_pitch_left_hand = calculate_average_pitch(left_hand_track)

    # map the notes to add a custom attribute on whether they come from left or right hand
    combined_notes = []
    for note in left_hand_track.notes:
        combined_notes.append((note, "left"))
    for note in right_hand_track.notes:
        combined_notes.append((note, "right"))
    # sort by start time and then by pitch descending
    combined_notes.sort(key=lambda n: (n[0].start, -n[0].pitch))

    melody_track_separated_new_right_hand = pretty_midi.Instrument(program=right_hand_track.program, is_drum=right_hand_track.is_drum, name="Right hand melody track")
    base_track_separated_new_left_hand = pretty_midi.Instrument(program=left_hand_track.program, is_drum=left_hand_track.is_drum, name="Left hand base track")

    last_note_in_melody_velocity = avg_velocity_right_hand
    last_note_in_melody_pitch = avg_pitch_right_hand
    last_note_in_melody_time = 0.0
    last_note_in_melody_sametime_count = 0

    last_note_in_non_melody_velocity = avg_velocity_left_hand
    last_note_in_non_melody_pitch = avg_pitch_left_hand
    last_note_in_non_melody_time = 0.0

    for note, hand in combined_notes:
        is_the_only_note_in_time = True
        
        highest_note_in_time = None
        lowest_note_in_time = None

        for other_note, other_hand in combined_notes:
            if other_note != note and note_is_starting_simultaneously(note, other_note):
                is_the_only_note_in_time = False
                if highest_note_in_time is None or other_note.pitch > highest_note_in_time.pitch:
                    highest_note_in_time = other_note
                if lowest_note_in_time is None or other_note.pitch < lowest_note_in_time.pitch:
                    lowest_note_in_time = other_note

        is_the_highest_note_in_time = (highest_note_in_time is None) or (note.pitch >= highest_note_in_time.pitch)
        is_the_lowest_note_in_time = (lowest_note_in_time is None) or (note.pitch <= lowest_note_in_time.pitch)

        is_close_in_velocity_to_melody_value = math.fabs(note.velocity - last_note_in_melody_velocity)
        is_close_in_velocity_to_melody = is_close_in_velocity_to_melody_value <= MAX_NOTE_VELOCITY_GAP_SAMEHAND
        is_close_in_pitch_to_melody_value = math.fabs(note.pitch - last_note_in_melody_pitch)
        is_close_in_pitch_to_melody = is_close_in_pitch_to_melody_value <= MAX_NOTE_PITCH_GAP_SAMEHAND
        is_close_in_time_to_melody_value = math.fabs(note.start - last_note_in_melody_time)
        is_close_in_time_to_melody = is_close_in_time_to_melody_value <= NOTE_CLOSE_IN_TIME_THRESHOLD

        is_the_only_note_in_time_weighted = MELODY_WEIGHT_ONLY_NOTE if is_the_only_note_in_time else 0.0
        is_the_highest_note_in_time_weighted = MELODY_WEIGHT_HIGHEST_NOTE if is_the_highest_note_in_time else 0.0
        is_close_in_velocity_to_melody_weighted = MELODY_WEIGHT_VELOCITY_SIMILARITY * (1.0 - (is_close_in_velocity_to_melody_value / MAX_NOTE_VELOCITY_GAP_SAMEHAND)) if is_close_in_velocity_to_melody else 0.0
        is_close_in_pitch_to_melody_weighted = MELODY_WEIGHT_PITCH_SIMILARITY * (1.0 - (is_close_in_pitch_to_melody_value / MAX_NOTE_PITCH_GAP_SAMEHAND)) if is_close_in_pitch_to_melody else 0.0
        is_close_in_time_to_melody_weighted = MELODY_WEIGHT_TIME_SIMILARITY * (1.0 - (is_close_in_time_to_melody_value / NOTE_CLOSE_IN_TIME_THRESHOLD)) if is_close_in_time_to_melody else 0.0

        melody_score = (
            is_the_only_note_in_time_weighted +
            is_the_highest_note_in_time_weighted +
            is_close_in_velocity_to_melody_weighted +
            is_close_in_pitch_to_melody_weighted +
            is_close_in_time_to_melody_weighted
        )

        is_close_in_velocity_to_non_melody_value = math.fabs(note.velocity - last_note_in_non_melody_velocity)
        is_close_in_velocity_to_non_melody = is_close_in_velocity_to_non_melody_value <= MAX_NOTE_VELOCITY_GAP_SAMEHAND
        is_close_in_pitch_to_non_melody_value = math.fabs(note.pitch - last_note_in_non_melody_pitch)
        is_close_in_pitch_to_non_melody = is_close_in_pitch_to_non_melody_value <= MAX_NOTE_PITCH_GAP_SAMEHAND
        is_close_in_time_to_non_melody_value = math.fabs(note.start - last_note_in_non_melody_time)
        is_close_in_time_to_non_melody = is_close_in_time_to_non_melody_value <= NOTE_CLOSE_IN_TIME_THRESHOLD
        is_close_in_velocity_to_non_melody_weighted = MELODY_WEIGHT_VELOCITY_SIMILARITY * (1.0 - (is_close_in_velocity_to_non_melody_value / MAX_NOTE_VELOCITY_GAP_SAMEHAND)) if is_close_in_velocity_to_non_melody else 0.0
        is_close_in_pitch_to_non_melody_weighted = MELODY_WEIGHT_PITCH_SIMILARITY * (1.0 - (is_close_in_pitch_to_non_melody_value / MAX_NOTE_PITCH_GAP_SAMEHAND)) if is_close_in_pitch_to_non_melody else 0.0
        is_close_in_time_to_non_melody_weighted = MELODY_WEIGHT_TIME_SIMILARITY * (1.0 - (is_close_in_time_to_non_melody_value / NOTE_CLOSE_IN_TIME_THRESHOLD)) if is_close_in_time_to_non_melody else 0.0

        non_melody_score = (
            is_close_in_velocity_to_non_melody_weighted +
            is_close_in_pitch_to_non_melody_weighted +
            is_close_in_time_to_non_melody_weighted
        )

        if not is_the_highest_note_in_time:
            melody_score -= MELODY_WEIGHT_NON_HIGHEST_NOTE_PENALTY  # penalize if not the highest note in simultaneous notes

        if is_the_lowest_note_in_time and not is_the_only_note_in_time:
            melody_score -= MELODY_WEIGHT_LOWEST_NOTE_PENALTY # penalize if it is the lowest note in simultaneous notes

        if (last_note_in_melody_pitch - note.pitch) >= MAX_NOTE_PITCH_GAP_SAMEHAND:
            melody_score -= MELODY_WEIGHT_LARGE_GAP_DOWN_PENALTY  # penalize large downward jumps in melody

        same_time_as_last_melody_note = math.fabs(note.start - last_note_in_melody_time) <= NOTE_SAME_TIME_THRESHOLD

        if same_time_as_last_melody_note and last_note_in_melody_sametime_count > MELODY_SAMETIME_NOTE_AMOUNT_PENALTY_STARTS:
            melody_score -= MELODY_SAMETIME_NOTE_AMOUNT_PENALTY

        if melody_score >= non_melody_score:
            # goes to melody track
            #if len(melody_track_separated_new_right_hand.notes) == 2:
            #    print(last_note_in_melody_pitch, last_note_in_melody_velocity, last_note_in_melody_time, last_note_in_melody_sametime_count)
            #    print(note.pitch, note.velocity, note.start)
            #    print(melody_score, non_melody_score)
            #    exit(1)
            melody_track_separated_new_right_hand.notes.append(note)
            if same_time_as_last_melody_note:
                last_note_in_melody_sametime_count += 1
            if is_the_highest_note_in_time or last_note_in_melody_pitch < note.pitch or not same_time_as_last_melody_note:
                last_note_in_melody_velocity = note.velocity
                last_note_in_melody_pitch = note.pitch
                last_note_in_melody_time = note.start
                last_note_in_melody_sametime_count = 1
        else:
            # goes to non-melody track
            base_track_separated_new_left_hand.notes.append(note)
            same_time_as_last_non_melody_note = math.fabs(note.start - last_note_in_non_melody_time) <= NOTE_SAME_TIME_THRESHOLD
            if is_the_highest_note_in_time or last_note_in_non_melody_pitch < note.pitch or not same_time_as_last_non_melody_note:
                last_note_in_non_melody_velocity = note.velocity
                last_note_in_non_melody_pitch = note.pitch
                last_note_in_non_melody_time = note.start

    return (base_track_separated_new_left_hand, melody_track_separated_new_right_hand)
        

def remove_echo_and_split_tracks(track: pretty_midi.Instrument) -> tuple[
    pretty_midi.Instrument,
    pretty_midi.Instrument,
    pretty_midi.Instrument,
    float,
]:
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

    # reorganize the notes in both tracks so the right track has the melody
    [left_hand_track, right_hand_track] = move_melody_to_right_hand(left_hand_track, right_hand_track)

    combined_track_with_echo = pretty_midi.Instrument(program=track.program, is_drum=False, name="Combined Track With Echo")
    combined_track_with_echo.notes = left_hand_track.notes + right_hand_track.notes
    combined_track_with_echo.notes.sort(key=lambda n: n.start)

    for track in [left_hand_track, right_hand_track]:
        remove_echo(track)

    # let's now find the note durations that are most common for the NOTE_SAME_TIME_THRESHOLD for the same start and end times
    notes_groups = []
    for track in [left_hand_track, right_hand_track]:
        for note in track.notes:
            # see if it fits in any existing group
            found = False
            # loop through existing groups to see if the note fits in any in the window given
            for note_group in notes_groups:
                for note_in_group in note_group:
                    if note_has_same_duration(note, note_in_group):
                        note_group.append(note)
                        found = True
                        break
                if found:
                    break
            # not found anywhere
            if not found:
                note_group = [note]
                notes_groups.append(note_group)

    note_groups_average_durations = [(sum((n.end - n.start) for n in group) / len(group), len(group)) for group in notes_groups if len(group) > 1]

    quantized_left_hand_track = pretty_midi.Instrument(program=left_hand_track.program, is_drum=left_hand_track.is_drum, name=left_hand_track.name)
    quantized_right_hand_track = pretty_midi.Instrument(program=right_hand_track.program, is_drum=right_hand_track.is_drum, name=right_hand_track.name)
    quantized_combined_track_with_echo = pretty_midi.Instrument(program=combined_track_with_echo.program, is_drum=combined_track_with_echo.is_drum, name=combined_track_with_echo.name)
    quantized_size = 0.0

    if len(note_groups_average_durations) > 0:
        most_common_min_duration = (min(note_groups_average_durations, key=lambda x: x[0])[0] / 2)
        quantized_size = round(most_common_min_duration, 3)
        # this will be our common duration to snap to and quantize
        # we start looping through notes and quantizing them in a grid
        for track in [left_hand_track, right_hand_track, combined_track_with_echo]:
            for note in track.notes:
                # round to 3 decimal places to avoid floating point issues
                quantized_start = round((note.start / most_common_min_duration) * most_common_min_duration, 3)
                quantized_end = round((note.end / most_common_min_duration) * most_common_min_duration, 3)
                if quantized_end <= quantized_start:
                    quantized_end = quantized_start + most_common_min_duration
                note_copy = pretty_midi.Note(
                    velocity=note.velocity,
                    pitch=note.pitch,
                    start=quantized_start,
                    end=quantized_end
                )
                
                if track == left_hand_track:
                    quantized_left_hand_track.notes.append(note_copy)
                elif track == right_hand_track:
                    quantized_right_hand_track.notes.append(note_copy)
                else:
                    quantized_combined_track_with_echo.notes.append(note_copy)

    most_common_note_duration = (max(note_groups_average_durations, key=lambda x: x[1])[0] / 2)

    return (quantized_left_hand_track, quantized_right_hand_track, quantized_combined_track_with_echo, quantized_size, most_common_note_duration)