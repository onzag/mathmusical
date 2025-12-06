import pretty_midi
import sys
sys.setrecursionlimit(3500)

from lib.util import display_set_of_notes, get_chord_name

def extract_high_melody(quantized_right_hand: pretty_midi.Instrument) -> pretty_midi.Instrument:
    """Extract the highest melody from the quantized right hand notes.

    Args:
        quantized_right_hand: A list of quantized notes in the right hand.
    Returns:
        A list of notes representing the highest melody.
    """

    highest_melody = pretty_midi.Instrument(program=quantized_right_hand.program, is_drum=False, name="H. Melody")
    time_to_highest_note = {}

    times_checked = set()

    for note in quantized_right_hand.notes:
        if note.start in times_checked:
            continue
        start_time = note.start
        other_notes_at_time = [note for note in quantized_right_hand.notes if note.start == start_time]
        higest_note = max(other_notes_at_time, key=lambda n: n.pitch)
        highest_melody.notes.append(higest_note)
        times_checked.add(start_time)

    return highest_melody

def extract_key_estimates(quantized_combined_track_with_echo: pretty_midi.Instrument) -> list[KeyEstimateGrouped]:
    """Extract chord notes from the quantized right hand notes.

    Args:
        quantized_right_hand: A list of quantized notes in the right hand.
        quantized_left_hand: A list of quantized notes in the left hand.
    Returns:
        A list of notes representing the chord notes.
    """

    # sort notes by start time and pitch, higher pitch first
    quantized_combined_track_with_echo.notes.sort(key=lambda n: (n.start, -n.pitch))
    
    grouped_notes = []
    current_time = None
    for note in quantized_combined_track_with_echo.notes:
        
        working_group = []
        if current_time is not None and note.start <= current_time:
            continue  # already processed this time

        current_time = note.start
        for other_note in quantized_combined_track_with_echo.notes:
            if other_note.start == note.start:
                working_group.append(other_note)
            # also add notes that are playing within this time frame
            elif other_note.start < note.start and other_note.end > note.start:
                working_group.append(other_note)
        grouped_notes.append((working_group, note.start))

    last_key_estimate = None
    for group, start_time in grouped_notes:
        last_key_estimate = KeyEstimate(group, start_time, last_key_estimate)

    return last_key_estimate.group_estimates()

def remove_microchords_artifacts(combined_chords_with_split: list[tuple[list[int], float]], qsize: float):
    # these microchords often occur because the midi file includes a swinging motion usually coming from stringed instruments
    # where the player moves quickly between two chords causing the note to rise/fall quickly, the program detects these as separate chords
    # and even a key change, which is not desired

    # we will remove chords that are smaller than 3 qsizes
    filtered_chords = []
    for i in range(len(combined_chords_with_split)):
        chord_notes, start_time = combined_chords_with_split[i]
        next_chord_start_time = None
        if i + 1 < len(combined_chords_with_split):
            next_chord_start_time = combined_chords_with_split[i + 1][1]

        if not next_chord_start_time:
            filtered_chords.append((chord_notes, start_time))
            continue

        duration = next_chord_start_time - start_time
        if duration >= qsize * 3.0:
            filtered_chords.append((chord_notes, start_time))
        else:
            # change the time of the next chord to be this chord's start time
            combined_chords_with_split[i + 1] = (combined_chords_with_split[i + 1][0], start_time)

    return filtered_chords

# created this method to experiment with extracting chords until the next intersection
# but somehow the original method worked better, so it is commented out so it works
# like the original method again, but leaving this here for future reference in case
# we want to experiment with it again
def extract_chords_until_next_intersection(
    all_potential_chords: list[tuple[list[set[int]], float]],
    start_index: int,
    accumulated_intersections: list[tuple[list[set[int]], float]],
    accumulating_intersection: list[tuple[set[int], float]] = None,
    intersection_space: list[tuple[set[int], float]] = None,
) -> tuple[tuple[list[set[int]], float], int]:
    # what we will do is that we will extract chords starting from the given start index
    # as if it was the first and then once we cannot use that same chord (or groups of chords) anymore, we will stop and return that chord along with the index we stopped at
    # each chord has its own fitness score, we will try to maximize that score
    chord_data_at_index = all_potential_chords[start_index]

    if intersection_space is not None:
        # check whether at least one chord in chord_data_at_index is in intersection_space
        has_intersection = False
        for chord_possible, chord_fitness in chord_data_at_index[0]:
            for intersection_chord, intersection_chord_fitness in intersection_space:
                if chord_possible == intersection_chord:
                    has_intersection = True
                    break
            if has_intersection:
                break

        if not has_intersection:
            # no intersection found, return None
            return None

    if start_index + 1 >= len(all_potential_chords):
        # we are at the end, return the current chord data
        return (chord_data_at_index, start_index + 1)
    
    # check if the next chord respects the intersection space before committing
    if intersection_space is not None:
        has_intersection = False
        next_chord_data = all_potential_chords[start_index + 1]
        for chord_possible, chord_fitness in next_chord_data[0]:
            for intersection_chord, intersection_chord_fitness in intersection_space:
                if chord_possible == intersection_chord:
                    has_intersection = True
                    break
            if has_intersection:
                break

        if not has_intersection:
            # no intersection found, return the current chord data
            return (chord_data_at_index, start_index + 1)

    new_intersection_space = intersection_space if intersection_space is not None else chord_data_at_index[0]
    new_accumulating_intersection = []
    # if accumulating_intersection is None:
    #     new_accumulating_intersection = chord_data_at_index[0]
    # else:
    #     for chord_possible, chord_fitness in accumulating_intersection:
    #         for chord_in_current, fitness_in_current in chord_data_at_index[0]:
    #             if chord_possible == chord_in_current:
    #                 new_accumulating_intersection.append((chord_possible, max(chord_fitness, fitness_in_current)))
    #                 break

    # if len(new_accumulating_intersection) == 0:
    #     # no intersection found, return the current chord data
    #     return (chord_data_at_index, start_index + 1)

    next_extraction = extract_chords_until_next_intersection(
        all_potential_chords,
        start_index + 1,
        accumulated_intersections,
        new_accumulating_intersection,
        new_intersection_space,
    )

    if next_extraction is None:
        # did not manage to respect the intersection_space, return the current chord data
        return (chord_data_at_index, start_index + 1)
    
    # calculate the intersection between chord_data_at_index and next_extraction[0]
    insersection_chords = []

    last_intersection = accumulated_intersections[-1] if len(accumulated_intersections) >= 1 else None
    previous_intersection_before_last = accumulated_intersections[-2] if len(accumulated_intersections) >= 2 else None

    # denying using negative basically means not using a chord with a negative fitness score
    # these are still valid but they come from using a note 
    has_been_going_as_far_as_previous_chord = False
    if previous_intersection_before_last is not None and last_intersection is not None:
        # we are going to check how long that intersection has been going on for
        previous_intersection_before_last_duration = abs(last_intersection[1] - previous_intersection_before_last[1])
        last_intersection_duration_thus_far = abs(chord_data_at_index[1] - last_intersection[1])
        if last_intersection_duration_thus_far >= previous_intersection_before_last_duration:
            # if the last intersection has been going on for at least as long as the one before it, we deny using negative scored chords
            has_been_going_as_far_as_previous_chord = True

    for chord_possible, chord_fitness in chord_data_at_index[0]:
        for next_chord_possible, next_chord_fitness in next_extraction[0][0]:
            if chord_possible == next_chord_possible:
                insersection_chords.append((chord_possible, max(chord_fitness, next_chord_fitness)))
                break

    if len(insersection_chords) == 0:
        # no further intersection found, return the current chord data
        return (chord_data_at_index, start_index + 1)
    else:
        max_intersection_chord_fitness = max((chord_fitness for chord_possible, chord_fitness in insersection_chords), default=None)
        max_current_accumulating_fitness = max((chord_fitness for chord_possible, chord_fitness in new_accumulating_intersection), default=None)
        # if max_intersection_chord_fitness > max_current_accumulating_fitness:
        #     return (chord_data_at_index, start_index + 1)
        return ((insersection_chords, chord_data_at_index[1]), next_extraction[1])


def extract_chords(quantized_combined_track_with_echo: pretty_midi.Instrument, key_estimates: list[KeyEstimateGrouped], microchord_artifact_size: float, most_common_note_duration: float) -> tuple[pretty_midi.Instrument, ...]:
    chord_instrument = pretty_midi.Instrument(program=quantized_combined_track_with_echo.program, is_drum=False, name="Chords")

    last_chord_end_time = 0.0
    last_chord_end_time = quantized_combined_track_with_echo.notes[-1].end if len(quantized_combined_track_with_echo.notes) > 0 else 0.0

    all_potential_chords = []
    for key_estimate in key_estimates:
        potential_chords = key_estimate.get_potential_chords_for_group()
        all_potential_chords += potential_chords

    intersection_so_far = []
    next_index = 0
    while next_index < len(all_potential_chords):
        (new_intersection, next_index) = extract_chords_until_next_intersection(all_potential_chords, next_index, intersection_so_far)
        intersection_so_far.append(new_intersection)

    final_chords = []
    for chords_entry in intersection_so_far:
        # pick the chord with the highest fitness
        best_chord = max(chords_entry[0], key=lambda x: x[1])
        # add it to the final chords with its start time
        final_chords.append((best_chord[0], chords_entry[1]))

    all_source_key_estimates = []
    for key_estimate in key_estimates:
        all_source_key_estimates += key_estimate.get_source_key_estimates()

    #final_chords.sort(key=lambda x: x[1])

    added_splits = []
    for source_estimate in all_source_key_estimates:
        if source_estimate.is_impactful_and_chordlike():
            chord_after_this_time = None
            for chord in final_chords:
                if chord[1] <= source_estimate.start_time:
                    chord_after_this_time = chord
                else:
                    break
                
            if chord_after_this_time is not None and chord_after_this_time[1] != source_estimate.start_time:
                # add a split chord here with the same notes as the chord after this time
                added_splits.append((chord_after_this_time[0], source_estimate.start_time))

    combined_chords_with_split = final_chords + added_splits
    combined_chords_with_split.sort(key=lambda x: x[1])

    combined_chords_with_split = remove_microchords_artifacts(combined_chords_with_split, microchord_artifact_size)

    lyrics = []

    for i in range(len(combined_chords_with_split)):
        chord_notes, start_time = combined_chords_with_split[i]
        next_chord_start_time = last_chord_end_time
        if i + 1 < len(combined_chords_with_split):
            next_chord_start_time = combined_chords_with_split[i + 1][1]

        # create a chord note for each note in the chord
        annotation = "^" + str(i + 1) + "|" + get_chord_name(chord_notes)

        for note in chord_notes:
            chord_note = pretty_midi.Note(
                velocity=40,
                pitch=note + 60 - 12,  # place chords in a mid range
                start=start_time,
                end=next_chord_start_time,  # fixed duration for now
            )
            chord_instrument.notes.append(chord_note)

        # add annotation as a lyric at the start time of the chord
        lyrics.append(pretty_midi.Lyric(text=annotation, time=start_time))

    for key_estimate in key_estimates:
        estimated_key = key_estimate.get_estimated_key()
        lyrics.append(pretty_midi.Lyric(text="K:" + str(estimated_key), time=key_estimate.start_time))

    return (chord_instrument, lyrics, combined_chords_with_split)

CHORD_SHAPES_PURE = [
    [0, 4, 7],    # Major triad
    [0, 3, 7],    # Minor triad
    [0, 3, 6],   # Diminished triad
    [0, 4, 8],   # Augmented triad,
]

CHORD_SHAPES_2_PURE = [
    [0, 7],    # Perfect fifth
    [0, 4],    # Major third
    [0, 3],    # Minor third
    [0, 6],    # Tritone
]

CHORD_SHAPES_WITH_VARIATIONS = CHORD_SHAPES_PURE  # Placeholder for future expansion
CHORD_SHAPES_2_WITH_VARIATIONS = CHORD_SHAPES_2_PURE
for i in range(1, 3):
    for shape in CHORD_SHAPES_PURE:
        note_to_displace = shape[i]
        new_shape = [(n - note_to_displace) % 12 for n in shape]
        shape_exists = False
        for existing_shape in CHORD_SHAPES_WITH_VARIATIONS:
            if set(existing_shape) == set(new_shape):
                shape_exists = True
                break
        if not shape_exists:
            CHORD_SHAPES_WITH_VARIATIONS.append(new_shape)
for i in range(1, 2):
    for shape in CHORD_SHAPES_2_PURE:
        note_to_displace = shape[i]
        new_shape = [(n - note_to_displace) % 12 for n in shape]
        shape_exists = False
        for existing_shape in CHORD_SHAPES_2_WITH_VARIATIONS:
            if set(existing_shape) == set(new_shape):
                shape_exists = True
                break
        if not shape_exists:
            CHORD_SHAPES_2_WITH_VARIATIONS.append(new_shape)

print("Total chord shapes: ", len(CHORD_SHAPES_WITH_VARIATIONS))
print("Total 2-note chord shapes: ", len(CHORD_SHAPES_2_WITH_VARIATIONS))

MASTER_SET_TO_KEY_SIGNATURE = {
    0: 0,
    -5: 1,
    2: 2,
    -3: 3,
    4: 4,
    -1: 5,
    6: 6,
    1: 7,
    -4: 8,
    3: 9,
    -2: 10,
    5: 11,
    -6: 12,
}

MASTER_SET_0 = [0,2,4,5,7,9,11]
MASTER_SETS = {}
MASTER_SETS_AS_SET = {}
for i in range(-7, 7):
    if i == 0:
        MASTER_SETS[i] = set(MASTER_SET_0)
        MASTER_SETS_AS_SET[i] = set(MASTER_SET_0)
        continue
    
    sign = 1 if i > 0 else -1
    changes = abs(i)
    start_point = 3 if sign == 1 else 6

    master_set_clone = [note for note in MASTER_SET_0]
    for j in range(changes):
        note_to_change = (start_point + (j * sign * 4)) % 7
        master_set_clone[note_to_change] += sign
        master_set_clone[note_to_change] = master_set_clone[note_to_change] % 12
    MASTER_SETS[i] = master_set_clone
    MASTER_SETS_AS_SET[i] = set(master_set_clone)

class KeyEstimateGrouped:
    def __init__(self, source_key_estimates: list[KeyEstimate]):
        self.source_key_estimates = source_key_estimates
        self.start_time = min(ke.start_time for ke in source_key_estimates)
        self.potential_master_set_ids = source_key_estimates[-1].potential_master_set_ids

        self.memoized_play_chords = {}

    def is_compatible(self, other: KeyEstimate) -> bool:
        # the valid master sets of the other estimate must intersect with our potential master sets
        return len(self.potential_master_set_ids.intersection(other.valid_master_sets_ids)) != 0
    
    def add_estimate(self, new_estimate: KeyEstimate) -> None:
        self.source_key_estimates.append(new_estimate)
        self.potential_master_set_ids = self.potential_master_set_ids.intersection(new_estimate.valid_master_sets_ids)

    def get_estimated_key(self) -> int:
        # return the lowest id from the absolute value of the potential master sets
        # that is the one that is closest to zero, remember there may be negative values
        return min(self.potential_master_set_ids, key=lambda x: abs(x))
    
    def get_notes_to_drop(self) -> list[int]:
        notes_to_drop = []
        for estimate in self.source_key_estimates:
            notes_to_drop_for_estimate = estimate.get_notes_to_drop(self.get_estimated_key())
            if len(notes_to_drop_for_estimate) > 0:
                notes_to_drop.append((notes_to_drop_for_estimate, estimate.start_time))
        return notes_to_drop
    
    def get_track_without_notes_to_drop(self, instrument: pretty_midi.Instrument) -> pretty_midi.Instrument:
        notes_to_drop = self.get_notes_to_drop()
        if len(notes_to_drop) == 0:
            return instrument  # nothing to drop
        modified_instrument = pretty_midi.Instrument(program=instrument.program, is_drum=instrument.is_drum, name=instrument.name)
        for note in instrument.notes:
            drop_note = False
            for notes_to_drop_set, start_time in notes_to_drop:
                if note.start == start_time and (note.pitch % 12) in notes_to_drop_set:
                    drop_note = True
                    break
            if not drop_note:
                modified_instrument.notes.append(note)
        return modified_instrument
    
    def get_signature(self) -> pretty_midi.KeySignature:
        estimated_key = self.get_estimated_key()
        return pretty_midi.KeySignature(MASTER_SET_TO_KEY_SIGNATURE[estimated_key], self.start_time)
    
    def get_potential_chords_for_group(self) -> list[tuple[list[set[int]], float]]:
        potential_chords = []
        estimated_key = self.get_estimated_key()

        for estimate in self.source_key_estimates:
            chords_for_estimate = estimate.get_potential_chords(estimated_key)
            if len(chords_for_estimate[0]) == 0:
                raise ValueError("No potential chords found for estimate at time " + str(estimate.start_time) + " with estimate " + str(estimate) + " in the master set " + str(estimated_key))
            potential_chords.append(chords_for_estimate)

        return potential_chords
    
    def can_play_chord_at_time(self, chord: set[int], start_time: float, end_time: float) -> bool:
        id = str(sorted(list(chord))) + "_" + str(start_time) + "_" + str(end_time)
        if id in self.memoized_play_chords:
            return self.memoized_play_chords[id]

        estimated_key = self.get_estimated_key()
        all_approve = True

        for estimate in self.source_key_estimates:
            if start_time <= estimate.start_time < end_time:
                chords_for_estimate, fitness = estimate.get_potential_chords(estimated_key)
                chord_is_valid = False
                for chord_possible, chord_fitness in chords_for_estimate:
                    if chord_possible == chord:
                        chord_is_valid = True
                        break
                if not chord_is_valid:
                    all_approve = False
                    break

        self.memoized_play_chords[id] = all_approve

        return all_approve
    
    def get_key_estimate_at_time(self, time: float, fuzzyByAmount: float = 0.0) -> KeyEstimate | None:
        matches = []
        for estimate in self.source_key_estimates:
            if abs(estimate.start_time - time) <= fuzzyByAmount:
                matches.append(estimate)
        if len(matches) > 0:
            # return the closest match
            return min(matches, key=lambda e: abs(e.start_time - time))
        return None
    
    def get_source_key_estimates(self) -> list[KeyEstimate]:
        return self.source_key_estimates
    
    def __repr__(self):
        return f"KeyEstimateGrouped(start_time={self.start_time}, potential_master_set_ids={display_set_of_notes(self.potential_master_set_ids)}, estimated_key={self.get_estimated_key()}, notes_to_drop={self.get_notes_to_drop()})"

class KeyEstimate:
    def __init__(
            self,
            reference_notes: list[pretty_midi.Note],
            start_time: float,
            previous_key_estimate: KeyEstimate | None,
        ):
        self.reference_notes_mod_12 = set(note.pitch % 12 for note in reference_notes)
        self.start_time = start_time
        self.notes = reference_notes

        self.valid_master_sets_ids = set([])
        self.previous_key_estimate = previous_key_estimate
        self.next_key_estimate = None

        self.previous_key_estimate.indicate_next(self) if self.previous_key_estimate is not None else None

        self.memoized_potential_chords = {}

        for key, master_set in MASTER_SETS_AS_SET.items():
            if self.reference_notes_mod_12.issubset(master_set):
                self.valid_master_sets_ids.add(key)

        if len(self.valid_master_sets_ids) == 0:
            # try dropping one note at a time to see if we can find a valid master set that is compatible
            print("No valid master sets found for the given reference notes, attempting to drop notes, " + str(self.reference_notes_mod_12))
            for note in list(self.reference_notes_mod_12):
                modified_reference_notes = self.reference_notes_mod_12.copy()
                modified_reference_notes.remove(note)
                for key, master_set in MASTER_SETS_AS_SET.items():
                    if modified_reference_notes.issubset(master_set):
                        self.valid_master_sets_ids.add(key)

        if len(self.valid_master_sets_ids) == 0:
            raise ValueError("No valid master sets found for the given reference notes, this song is unclassifiable, " + str(self.reference_notes_mod_12))

        self.potential_master_set_ids = self.valid_master_sets_ids.copy()
        
        intersection_calculation = self.valid_master_sets_ids.copy()
        current_estimate_to_check = self.previous_key_estimate
        while current_estimate_to_check is not None:
            # calculate the intersection between our current potential master sets and the previous chord's potential master sets
            intersection_calculation = intersection_calculation.intersection(current_estimate_to_check.valid_master_sets_ids)
            if len(intersection_calculation) > 0:
                current_estimate_to_check.potential_master_set_ids = intersection_calculation
                self.potential_master_set_ids = intersection_calculation
            else:
                break  # no more intersection possible, exit early
            current_estimate_to_check = current_estimate_to_check.previous_key_estimate

    def indicate_next(self, next_estimate: KeyEstimate) -> None:
        self.next_key_estimate = next_estimate

    def group_estimates(self) -> list[KeyEstimateGrouped]:
        if not self.previous_key_estimate:
            return [KeyEstimateGrouped([self])]
        
        previous_grouped_estimates = self.previous_key_estimate.group_estimates()
        last_estimate = previous_grouped_estimates[-1]
        if last_estimate.is_compatible(self):
            last_estimate.add_estimate(self)
            return previous_grouped_estimates
        else:
            return previous_grouped_estimates + [KeyEstimateGrouped([self])]
        
    def get_notes_to_drop(self, expected_master_set_id: int | None = None) -> list[int]:
        # return the notes in our mod_12 that are not in any of our potential master sets
        notes_to_drop = []
        for note in self.reference_notes_mod_12:
            master_set = MASTER_SETS_AS_SET[expected_master_set_id]
            if note not in master_set:
                notes_to_drop.append(note)
        amount_of_dropped_notes = len(notes_to_drop)
        current_notes_amount = len(self.reference_notes_mod_12)
        
        # if even after dropping we have 5 notes playing at the same time,
        # the sound is likely dissonant and not a good chord
        if (current_notes_amount - amount_of_dropped_notes) >= 5:
            solved_5_issue = False
            reference_notes_without_drops = self.reference_notes_mod_12.difference(set(notes_to_drop))

            # we will try to fix this by dropping the least important notes
            # for that we will form chords using CHORD_SHAPES_WITH_VARIATIONS until
            # we find one that fits and drop one of the other two notes left
            for note in reference_notes_without_drops:
                for shape_def in CHORD_SHAPES_WITH_VARIATIONS:
                    chord_shape = set()
                    invalid_shape = False

                    # check each distance in the shape definition
                    for distance in shape_def:
                        # calculate the note to add
                        note_to_add = (note + distance) % 12
                        # check if this note is in the master set and in our reference notes
                        if note_to_add in MASTER_SETS_AS_SET[expected_master_set_id]:
                            chord_shape.add(note_to_add)
                        else:
                            invalid_shape = True
                            break
                    if not invalid_shape:
                        # check that every note in the chord shape is in our reference notes
                        if chord_shape.issubset(reference_notes_without_drops):
                            # now get the two notes that are not in the chord shape
                            notes_not_in_chord_shape = reference_notes_without_drops.difference(chord_shape)
                            # drop one of them (the first one)
                            note_to_drop = list(notes_not_in_chord_shape)[0]
                            notes_to_drop.append(note_to_drop)
                            solved_5_issue = True
                            break
                if solved_5_issue:
                    break

            if not solved_5_issue:
                # as a last resort, drop the highest note
                highest_note = max(reference_notes_without_drops)
                notes_to_drop.append(highest_note)
                        
        return notes_to_drop
    
    def is_impactful_and_chordlike(self):
        notes_that_start_at_this_time = [note for note in self.notes if note.start == self.start_time]
        max_note_end_time = max(note.end for note in notes_that_start_at_this_time)  # small leeway for notes that start just before
        is_too_short = max_note_end_time - self.start_time < 0.5
        is_very_long = max_note_end_time - self.start_time > 2.5

        if is_very_long and len(notes_that_start_at_this_time) >= 2:
            return True

        if is_too_short:
            return False

        # just count the notes in the reference notes
        if len(self.reference_notes_mod_12) >= 3:
            return True
        
        # count it in the general notes
        if len(notes_that_start_at_this_time) >= 3:
            # lets now check if the duration of this keyestimate bleeds into the next one
            
            if self.next_key_estimate is not None:
                if max_note_end_time > self.next_key_estimate.start_time + 0.05:  # small leeway
                    return True
                
            # otherwise let's just see if it is longer than 0.5 seconds
            if max_note_end_time - self.start_time >= 0.5:
                return True

        return False
    
    def get_potential_chords(self, master_set_id: int, remove_echoing_notes: bool = False, two_note_chords: bool = False) -> tuple[list[set[int]], float]:
        id = str(master_set_id) + "_" + str(remove_echoing_notes) + "_" + str(two_note_chords)
        if id in self.memoized_potential_chords:
            return self.memoized_potential_chords[id]
        
        notes_to_drop = self.get_notes_to_drop(master_set_id)
        reference_notes_without_drops = self.reference_notes_mod_12.difference(set(notes_to_drop))

        if remove_echoing_notes:
            for note in self.notes:
                # check if this note is an echoing note
                # basically it starts before our start time
                is_echoing = note.start < self.start_time
                if is_echoing:
                    note_mod_12 = note.pitch % 12
                    if note_mod_12 in reference_notes_without_drops:
                        reference_notes_without_drops.remove(note_mod_12)

        if len(reference_notes_without_drops) <= 1:
            # single notes are compatible with every chord of the same master set
            all_notes = MASTER_SETS_AS_SET[master_set_id]
            all_chords = []
            for note in all_notes:
                for shape_def in CHORD_SHAPES_PURE + CHORD_SHAPES_2_PURE:
                    chord_shape = set()

                    # check each distance in the shape definition
                    for distance in shape_def:
                        # calculate the note to add
                        note_to_add = (note + distance) % 12
                        chord_shape.add(note_to_add)

                    all_chords.append(chord_shape)
            result = None
            if len(reference_notes_without_drops) == 1:
                # if the note is in the chord, give it a fitness of 3, if it doesn't give it a fitness of 1
                note_in_question = list(reference_notes_without_drops)[0]
                result = ([(p, 3) if note_in_question in p else (p, 1) for p in all_chords], self.start_time)
            else:
                result = ([(p, 1) for p in all_chords], self.start_time) 
            self.memoized_potential_chords[id] = result
            return result

        SET_FOR_VARIATIONS = CHORD_SHAPES_WITH_VARIATIONS if not two_note_chords else CHORD_SHAPES_2_WITH_VARIATIONS
        SET_FOR_PERFECT = CHORD_SHAPES_PURE if not two_note_chords else CHORD_SHAPES_2_PURE
        
        potential_chords = []
        potential_chords_fitness = []
        perfect_chords_found = []
        # first try all chord shapes starting from each note in the reference notes
        for note in reference_notes_without_drops:
            # try each chord shape
            for shape_def in SET_FOR_VARIATIONS:
                # build the chord shape starting from this note
                chord_shape = set()
                invalid_shape = False

                is_pure_shape = shape_def in SET_FOR_PERFECT

                # check each distance in the shape definition
                for distance in shape_def:
                    # calculate the note to add
                    note_to_add = (note + distance) % 12
                    # check if this note is in the master set and in our reference notes
                    if note_to_add in MASTER_SETS_AS_SET[master_set_id]:
                        chord_shape.add(note_to_add)
                    else:
                        invalid_shape = True
                        break
                dissonant_sound = False
                if not invalid_shape:
                    # we are going to check for a dissonant sound, basically 5 notes playing at the same time all things considered
                    set_of_playing_notes = reference_notes_without_drops.union(chord_shape)
                    if len(set_of_playing_notes) >= 5:
                        dissonant_sound = True
                if not invalid_shape and not dissonant_sound:
                    potential_chords.append(chord_shape)

                    # check if the chord shape is a subset of our reference notes
                    if chord_shape.issubset(reference_notes_without_drops):
                        perfect_chords_found.append(chord_shape)

                    # check how many notes from the chord shape are in our reference notes
                    notes_in_shape_and_reference = chord_shape.intersection(reference_notes_without_drops)
                    fitness = len(notes_in_shape_and_reference)
                    potential_chords_fitness.append(fitness)
                    if not is_pure_shape:
                        # penalize non pure shapes
                        # ensure they are negative
                        potential_chords_fitness[-1] -= 4

        if len(perfect_chords_found) > 0:
            result = ([(p, 4) for p in perfect_chords_found], self.start_time)  # return only perfect chords found
            self.memoized_potential_chords[id] = result
            return result
        
        to_return = (list(zip(potential_chords, potential_chords_fitness)), self.start_time)  # return all potential chords found
        if len(to_return[0]) == 0:
            if not remove_echoing_notes:
                to_return = self.get_potential_chords(master_set_id, remove_echoing_notes=True, two_note_chords=two_note_chords)  # try again removing echoing notes
            elif not two_note_chords:
                to_return = self.get_potential_chords(master_set_id, remove_echoing_notes=remove_echoing_notes, two_note_chords=True)  # try again allowing two note chords
        self.memoized_potential_chords[id] = to_return
        return to_return

        
    def __repr__(self):
        return f"KeyEstimate(start_time={self.start_time}, reference_notes_mod_12={display_set_of_notes(self.reference_notes_mod_12)}, valid_master_sets_ids={self.valid_master_sets_ids}, potential_master_set_ids={self.potential_master_set_ids})"

def extract_rythm(quantized_left_hand: pretty_midi.Instrument, quantized_right_hand: pretty_midi.Instrument, qsize: float) -> pretty_midi.Instrument:
    combined_notes = quantized_right_hand.notes + quantized_left_hand.notes
    # sort notes by start time and pitch, higher pitch first
    combined_notes.sort(key=lambda n: (n.start, -n.pitch))

    # new ordered set of times
    rythm_times = []
    for note in combined_notes:
        if len(rythm_times) == 0 or note.start != rythm_times[-1]['start']:
            rythm_times.append({
                'start': note.start,
                'count': 1,
                'average_duration': note.end - note.start,
                'longest_duration': note.end - note.start,
            })
        else:
            rythm_times[-1] = {
                'start': rythm_times[-1]['start'],
                'count': rythm_times[-1]['count'] + 1,
                'average_duration': (rythm_times[-1]['average_duration'] * rythm_times[-1]['count'] + (note.end - note.start)) / (rythm_times[-1]['count'] + 1),
                'longest_duration': max(rythm_times[-1]['longest_duration'], note.end - note.start),
            }

    most_common_note_difference = 0.0
    grid_size = 0.0

    # make a map of numpy floats to counts
    difference_count = {}
    for note in combined_notes:
        comparison_note_starts = set([n.start for n in combined_notes if n.start >= note.start + (qsize * 2.0) and n.start <= note.end + (qsize * 10.0)])  # only consider notes that are at least 2 qsize away
        for next_note_start in comparison_note_starts:
            difference = abs(next_note_start - note.start)
            if difference not in difference_count:
                # before adding the note, see if there is a duration which is either double or half of this duration, and if so, increase its count instead
                # giving a leeway for note durations that are slightly off
                added_as_existing = False
                for existing_difference in difference_count.keys():
                    is_half = abs(existing_difference - (difference / 2)) <= 0.01
                    is_double = abs(existing_difference - (difference * 2)) <= 0.01
                    if is_half:
                        difference_count[difference] = difference_count[existing_difference] + 1
                        del difference_count[existing_difference]
                        added_as_existing = True
                        break
                    if is_double:
                        difference_count[existing_difference] += 1
                        added_as_existing = True
                        break
                if not added_as_existing:
                    difference_count[difference] = 0
                else:
                    continue
            difference_count[difference] += 1

    if len(difference_count) > 0:
        most_common_note_difference = None
        for difference, count in difference_count.items():
            if most_common_note_difference is None or count > difference_count[most_common_note_difference]:
                most_common_note_difference = difference

    print(f"Most common note difference: {most_common_note_difference}")
    grid_size = most_common_note_difference / 4.0  # subdivide into 4 for more precision
    # make the grid_size a multiple of qsize
    grid_size = round(grid_size / qsize) * qsize

    print(f"Determined grid size: {grid_size} (most common note difference: {most_common_note_difference})")

    current_grid_point = 0.0
    current_grid_size = round((grid_size * 2.0) / qsize) * qsize  # start with double grid size to allow for initial tempo changes
    current_grid_size_change_leeway = qsize  # allow grid size changes within this leeway

    grid_ends_at = rythm_times[-1]['start'] + rythm_times[-1]['longest_duration'] + current_grid_size

    # pick the first note start as the grid start point
    if len(rythm_times) > 0:
        current_grid_point = rythm_times[0]['start']

    grid_points = [current_grid_point]
    grid_point_had_reference_note = [True]
    grid_point_changed_grid_size = [False]
    grid_point_accuracies = [0.75]
    grid_point_grid_changes = [0]
    grid_point_score_thus_far = [0.75]
    grid_pivotal_point = [False]

    print(f"Leeway for grid size changes: {current_grid_size_change_leeway}")
    print(f"Qsize is: {qsize}")
    print(f"Current grid size: {current_grid_size}")

    while current_grid_point < grid_ends_at:
        next_grid_point = current_grid_point + current_grid_size
        # find a note that is within the leeway of the next grid point
        found_notes = []
        for rythm_time in rythm_times:
            if abs(rythm_time['start'] - next_grid_point) <= current_grid_size_change_leeway:
                found_notes.append(rythm_time)
        # find the one closest to zero
        if len(found_notes) > 0:
            closest_note = min(found_notes, key=lambda x: abs(x['start'] - next_grid_point))
            grid_points.append(closest_note['start'])
            grid_point_had_reference_note.append(True)
            
            # update the current grid size to be the distance to the closest note
            # so that we can adapt to tempo changes
            new_grid_size = round(abs(closest_note['start'] - current_grid_point) / qsize) * qsize
            if (new_grid_size == 0.0):
                # found itself, keep current grid size
                new_grid_size = current_grid_size
            
            if new_grid_size != current_grid_size:
                current_grid_size = new_grid_size
                grid_point_changed_grid_size.append(True)
            else:
                grid_point_changed_grid_size.append(False)
            current_grid_point = closest_note['start']
        else:
            grid_points.append(next_grid_point)
            grid_point_had_reference_note.append(False)
            grid_point_changed_grid_size.append(False)
            current_grid_point = next_grid_point

        # evaluate how we are doing so far
        # maximize this
        accuracy_notes_hit = sum((1 for had_note in grid_point_had_reference_note if had_note))
        # minimize this
        grid_changes_hit = sum((1 for changed in grid_point_changed_grid_size if changed))
        accuracy = accuracy_notes_hit / len(grid_point_had_reference_note)
        grid_changes = grid_changes_hit / len(grid_point_changed_grid_size)
        grid_point_accuracies.append(accuracy)
        grid_point_grid_changes.append(grid_changes)

        score = accuracy - (grid_changes/2.0)

        grid_point_score_thus_far.append(score)

        if score >= 0.8:
            grid_pivotal_point.append(True)
        else:
            grid_pivotal_point.append(False)
                

    grid_instrument = pretty_midi.Instrument(program=quantized_right_hand.program, is_drum=True, name="Rythm")

    print("score progression:" + str(grid_point_score_thus_far))
    print("pivotal points:" + str(grid_pivotal_point))

    for i in range(len(grid_points) - 1):
        grid_note = pretty_midi.Note(
            velocity=100,
            pitch=35 if grid_point_had_reference_note[i] else 36,  # Acoustic Bass Drum or Bass Drum 1
            start=grid_points[i],
            end=grid_points[i] + 0.1,
        )
        grid_instrument.notes.append(grid_note)
    
    return grid_instrument