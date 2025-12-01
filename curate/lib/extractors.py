import pretty_midi

from lib.util import display_set_of_notes

CHORDS_COMMON_TIME_STRUCTURES_3 = []
CHORDS_COMMON_TIME_STRUCTURES_4 = []

def merge_combinations(arr):
    results = set()
    def helper(current):
        results.add(tuple(current))
        for i in range(len(current) - 1):
            merged = current[:i] + [current[i] + current[i+1]] + current[i+2:]
            helper(merged)
    helper(arr)
    return [list(r) for r in results if len(r) < len(arr) or r == tuple(arr)]

# calculate chord time structures for that we start with a whole
for set_n in range(2,4):
    for divide_times in range(1, 4):
        if set_n == 3 and divide_times == 3:
            # too much complexity
            continue
        fractional_part = set_n**divide_times
        time_structure = [1/fractional_part]*fractional_part
        # now we create time structures

        all_combinations = merge_combinations(time_structure)

        for combination in all_combinations:
            if len(combination) == 1:
                continue  # 1 is too basic to matter
            if set_n == 2:
                # check if the element does not exist already
                exists = False
                for existing in CHORDS_COMMON_TIME_STRUCTURES_4:
                    if existing == combination:
                        exists = True
                        break
                if not exists:
                    CHORDS_COMMON_TIME_STRUCTURES_4.append(combination)
            else:
                # check if the element does not exist already
                exists = False
                for existing in CHORDS_COMMON_TIME_STRUCTURES_3:
                    if existing == combination:
                        exists = True
                        break
                if not exists:
                    CHORDS_COMMON_TIME_STRUCTURES_3.append(combination)

CHORDS_COMMON_TIME_STRUCTURES_GENERAL = CHORDS_COMMON_TIME_STRUCTURES_3 + CHORDS_COMMON_TIME_STRUCTURES_4

# sort all by largest to smallest
CHORDS_COMMON_TIME_STRUCTURES_GENERAL.sort(key=lambda x: -len(x))
CHORDS_COMMON_TIME_STRUCTURES_3.sort(key=lambda x: -len(x))
CHORDS_COMMON_TIME_STRUCTURES_4.sort(key=lambda x: -len(x))

print("Total common time structures: ", len(CHORDS_COMMON_TIME_STRUCTURES_GENERAL))

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

def calculate_common_chord_structures(
        key_estimates: list[KeyEstimateGrouped],
        combined_chords_with_split: list[tuple[list[int], float]],
        quantized_combined_track_with_echo: pretty_midi.Instrument,
        qsize: float,
        start_index: int = 0,
        use_reverse: bool = False,
    ) -> tuple[list[tuple[list[int], float]], list[tuple[list[float], list[int], list[tuple[float, float]], int, int]], int]:
    # structure of weights list, inner list is a list of indexes where the structure was found, new start and end time, the type (3 or 4) and a score

    track_ends_time = quantized_combined_track_with_echo.notes[-1].end if len(quantized_combined_track_with_echo.notes) > 0 else 0.0

    # we will start by analyzing this list of chords to find common time structures
    structures_found = [] # containst the structure that was found, the list of indexes where it was found, whether it is a 3-type or 4-type structure and a score for how good it is
    already_processed_indexes = []
    for time_structure in [[0.5,0.5]]:# in CHORDS_COMMON_TIME_STRUCTURES_GENERAL:
        group_size = len(time_structure)

        range_to_use = range(start_index, len(combined_chords_with_split) - group_size + 1)
        if use_reverse:
            range_to_use = range(len(combined_chords_with_split) - group_size - start_index, -1, -1)
        for i in range_to_use:
            if i in already_processed_indexes:
                continue  # already part of a found structure
            indexes_of_potential_structure = list(range(i, i + group_size))
            # check if the time differences match the time structure
            is_end_chord = i + group_size >= len(combined_chords_with_split) # an end chord may be longer than expected

            all_durations: list[float] = []
            actual_chord_start_and_end_times: list[tuple[float, float]] = []
            if is_end_chord:
                last_chord_duration_for_calculation = track_ends_time - combined_chords_with_split[indexes_of_potential_structure[-1]][1]
                # we will use the maximum time of another chord that plays that is longest provided it is not the end chord itself and provided such value is shorter than itself
                for index in indexes_of_potential_structure[:-1]:
                    chord_start_time = combined_chords_with_split[index][1]
                    next_chord_start_time = combined_chords_with_split[index + 1][1]
                    duration = next_chord_start_time - chord_start_time
                    all_durations.append(duration)
                duration_to_consider = max(all_durations)
                if duration_to_consider < last_chord_duration_for_calculation:
                    last_chord_duration_for_calculation = duration_to_consider
                all_durations.append(last_chord_duration_for_calculation)
                actual_chord_start_and_end_times = [
                    (
                        combined_chords_with_split[index][1],
                        combined_chords_with_split[index + 1][1] if index + 1 < len(combined_chords_with_split) else track_ends_time
                    ) for index in indexes_of_potential_structure
                ]
            else:
                for index in indexes_of_potential_structure:
                    chord_start_time = combined_chords_with_split[index][1]
                    next_chord_start_time = combined_chords_with_split[index + 1][1]
                    duration = next_chord_start_time - chord_start_time
                    all_durations.append(duration)
                    actual_chord_start_and_end_times.append((chord_start_time, next_chord_start_time))

            entire_duration = sum(all_durations)
            # now we calculate the ratios of these durations
            duration_ratios = [d / entire_duration for d in all_durations]
            # and then we see if it matches our time structure, giving some leeway
            matches_structure_exactly = True
            for j in range(len(time_structure)):
                expected_ratio = time_structure[j]
                actual_ratio = duration_ratios[j]
                if abs(expected_ratio - actual_ratio) > 0.05 * expected_ratio:
                    matches_structure_exactly = False
                    break

            new_chord_start_and_end_times = actual_chord_start_and_end_times

            matches_structure_with_adjustment = True
            adjustment_rate = 0.0
            if not matches_structure_exactly:
                # we will try to see if the structure can be forced to see if it fits
                potential_chord_start_and_end_times = actual_chord_start_and_end_times.copy()
                for i in range(len(time_structure)):
                    expected_ratio = time_structure[i]
                    expected_duration = expected_ratio * entire_duration
                    expected_start_time = combined_chords_with_split[indexes_of_potential_structure[0]][1] + sum(
                        time_structure[k] * entire_duration for k in range(i)
                    )
                    expected_end_time = expected_start_time + expected_duration
                    # quantize using qsize

                    expected_start_time_quantized = round(expected_start_time / qsize) * qsize
                    expected_end_time_quantized = round(expected_end_time / qsize) * qsize
                    expected_chord = combined_chords_with_split[indexes_of_potential_structure[i]][0]

                    potential_chord_start_and_end_times[i] = (expected_start_time_quantized, expected_end_time_quantized)

                    all_approve = True
                    for key_estimate in key_estimates:
                        this_approves = key_estimate.can_play_chord_at_time(set(expected_chord), expected_start_time_quantized, expected_end_time_quantized)
                        if not this_approves:
                            all_approve = False
                            break

                    if all_approve:
                        # get the key estimate at a given time to see if there is a note playing into it
                        # we should attempt to snap the time to the nearest estimate time within a fuzzy range
                        # to allow for human timing errors
                        key_estimate_at_time = None
                        for key_estimate_group in key_estimates:
                            key_estimate_at_time = key_estimate_group.get_key_estimate_at_time(
                                combined_chords_with_split[indexes_of_potential_structure[i]][1],
                                fuzzyByAmount=qsize*2.0,
                            )
                            if key_estimate_at_time is not None:
                                break

                        if key_estimate_at_time is not None:
                            # update the potential chord times to reflect this
                            potential_chord_start_and_end_times[i] = (key_estimate_at_time.start_time, expected_end_time_quantized)
                        else:
                            # we are going to check if it is a zone of big silence before saying it can't be there
                            # for that we simply are going to grab the nearest key estimate by using a bigger fuzzy amount
                            key_estimate_closest = None
                            for key_estimate_group in key_estimates:
                                key_estimate_closest = key_estimates.get_key_estimate_closest(
                                    combined_chords_with_split[indexes_of_potential_structure[i]][1],
                                    fuzzyByAmount=1.0, #one second
                                )
                                if key_estimate_closest is not None:
                                    break
                            if key_estimate_closest is None:
                                # we assume silence zone hence the chord is allowed
                                pass
                            else:
                                # no silence zone, we do not approve
                                all_approve = False

                    if not all_approve:
                        matches_structure_with_adjustment = False
                        break

                if matches_structure_with_adjustment:
                    # calculate the adjustment rate comparing the time_structure durations to the actual durations
                    total_adjustment = 0.0
                    # this should be a number between 0 and 1
                    for j in range(len(time_structure)):
                        expected_ratio = time_structure[j]
                        actual_ratio = duration_ratios[j]
                        total_adjustment += abs(expected_ratio - actual_ratio) / expected_ratio
                    adjustment_rate = total_adjustment / len(time_structure)
                    new_chord_start_and_end_times = potential_chord_start_and_end_times
    
            score_for_structure = 0

            if matches_structure_exactly:
                score_for_structure += 10  # perfect match

            if matches_structure_with_adjustment:
                score_for_structure += 5  # good match with adjustment
                # use the adjustment rate to decrease the score, higher adjustment rate means lower score
                score_for_structure += int((1.0 - adjustment_rate) * 5)

            # check if the time structure is a symmetric list
            if time_structure == time_structure[::-1] and score_for_structure > 0:
                score_for_structure += 5  # symmetric structures are more likely

            if score_for_structure > 0:
                # we found a perfectly matching structure
                structures_found.append((
                    time_structure,
                    indexes_of_potential_structure,
                    new_chord_start_and_end_times,
                    3 if time_structure in CHORDS_COMMON_TIME_STRUCTURES_3 else 4,
                    score_for_structure,
                ))
                already_processed_indexes += indexes_of_potential_structure
            

    results = combined_chords_with_split
    negative_score = 0

    # count the unmatched indexes and make that be the negative score as a negative number
    for i in range(len(combined_chords_with_split)):
        if i not in already_processed_indexes:
            negative_score -= 1

    if start_index == 0 and not use_reverse:
        result_from_start_1 = calculate_common_chord_structures(key_estimates, combined_chords_with_split, quantized_combined_track_with_echo, qsize, start_index=1, use_reverse=False)
        result_from_start_2 = calculate_common_chord_structures(key_estimates, combined_chords_with_split, quantized_combined_track_with_echo, qsize, start_index=2, use_reverse=False)
        result_from_start_3 = calculate_common_chord_structures(key_estimates, combined_chords_with_split, quantized_combined_track_with_echo, qsize, start_index=3, use_reverse=False)
        result_from_start_0_rev = calculate_common_chord_structures(key_estimates, combined_chords_with_split, quantized_combined_track_with_echo, qsize, start_index=0, use_reverse=True)
        result_from_start_1_rev = calculate_common_chord_structures(key_estimates, combined_chords_with_split, quantized_combined_track_with_echo, qsize, start_index=1, use_reverse=True)
        result_from_start_2_rev = calculate_common_chord_structures(key_estimates, combined_chords_with_split, quantized_combined_track_with_echo, qsize, start_index=2, use_reverse=True)
        result_from_start_3_rev = calculate_common_chord_structures(key_estimates, combined_chords_with_split, quantized_combined_track_with_echo, qsize, start_index=3, use_reverse=True)

        all_results = [
            results,
            result_from_start_1,
            result_from_start_2,
            result_from_start_3,
            result_from_start_0_rev,
            result_from_start_1_rev,
            result_from_start_2_rev,
            result_from_start_3_rev,
        ]
        
        # TODO combine all the results so that the sum of negative scores is minimized, while the number of structures found of high score is maximized provided
        # that they are not overlapping in indexes

        structure_with_best_match = structures_found # we will just use the current one for now

        # now we will apply the changes from the structures found in the best result
        for structure in structure_with_best_match:
            time_structure, indexes_of_potential_structure, new_chord_start_and_end_times, structure_type, score_for_structure = structure
            for j in range(len(indexes_of_potential_structure)):
                index = indexes_of_potential_structure[j]
                new_start_time, new_end_time = new_chord_start_and_end_times[j]
                chord_notes = combined_chords_with_split[index][0]
                results[index] = (chord_notes, new_start_time)

    print(structures_found, negative_score)
    return (results, structures_found, negative_score)  # placeholder for now

def extract_chords(quantized_combined_track_with_echo: pretty_midi.Instrument, key_estimates: list[KeyEstimateGrouped], qsize: float) -> pretty_midi.Instrument:
    chord_instrument = pretty_midi.Instrument(program=quantized_combined_track_with_echo.program, is_drum=False, name="Chords")

    last_chord_end_time = 0.0
    last_chord_end_time = quantized_combined_track_with_echo.notes[-1].end if len(quantized_combined_track_with_echo.notes) > 0 else 0.0

    all_potential_chords = []
    for key_estimate in key_estimates:
        potential_chords = key_estimate.get_potential_chords_for_group()
        all_potential_chords += potential_chords

    # now we want to loop from the start to the end, and for each estimate, we want to keep only the chords that are compatible with the previous estimate's chords
    # we want to calculate the intersection
    intersection_so_far = []
        
    for chords_for_estimate in all_potential_chords:
        if len(intersection_so_far) == 0:
            intersection_so_far.append(chords_for_estimate)
        else:
            last_intersection = intersection_so_far[-1]
            previous_intersection_before_last = intersection_so_far[-2] if len(intersection_so_far) >= 2 else None
            deny_using_negative_scored_chords = False
            if previous_intersection_before_last is not None:
                # we are going to check how long that intersection has been going on for
                previous_intersection_before_last_duration = abs(last_intersection[1] - previous_intersection_before_last[1])
                last_intersection_duration_thus_far = abs(chords_for_estimate[1] - last_intersection[1])
                if last_intersection_duration_thus_far >= previous_intersection_before_last_duration:
                    # if the last intersection has been going on for at least as long as the one before it, we deny using negative scored chords
                    deny_using_negative_scored_chords = True

            # check if last_intersection contains any chord that is compatible with any chord in the current chords_for_estimate
            new_intersection = []
            for last_chord_possible, last_chord_fitness in last_intersection[0]:
                # otherwise check for compatibility
                for current_chord_possible, current_chord_fitness in chords_for_estimate[0]:
                    # if we find a match, we add it to the new intersection
                    if last_chord_possible == current_chord_possible and (not deny_using_negative_scored_chords or current_chord_fitness >= 0):
                        # check that the new_chord does not already exist in the new_intersection
                        already_exists_at_index = -1
                        for i in range(len(new_intersection)):
                            if new_intersection[i][0] == last_chord_possible:
                                already_exists_at_index = i
                                break
                        if already_exists_at_index == -1:
                            new_intersection.append((last_chord_possible, max(last_chord_fitness, current_chord_fitness)))
                        else:
                            # update the fitness if the new one is higher
                            if new_intersection[already_exists_at_index][1] < max(last_chord_fitness, current_chord_fitness):
                                new_intersection[already_exists_at_index] = (last_chord_possible, max(last_chord_fitness, current_chord_fitness))
                        break

            if len(new_intersection) == 0:
                # no intersection found, we add a new entry
                intersection_so_far.append(chords_for_estimate)
            else:
                intersection_so_far[-1] = (new_intersection, last_intersection[1])  # update in place

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

    remove_microchords_artifacts(combined_chords_with_split, qsize)

    # now we want to find common shapes and time structures to make chords more consistent if such is possible
    combined_chords_with_split_final = calculate_common_chord_structures(key_estimates, combined_chords_with_split, quantized_combined_track_with_echo, qsize)

    for i in range(len(combined_chords_with_split)):
        chord_notes, start_time = combined_chords_with_split[i]
        next_chord_start_time = last_chord_end_time
        if i + 1 < len(combined_chords_with_split):
            next_chord_start_time = combined_chords_with_split[i + 1][1]

        # create a chord note for each note in the chord

        for note in chord_notes:
            chord_note = pretty_midi.Note(
                velocity=40,
                pitch=note + 60 - 12,  # place chords in a mid range
                start=start_time,
                end=next_chord_start_time,  # fixed duration for now
            )
            chord_instrument.notes.append(chord_note)

    return chord_instrument

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
    
    def get_potential_chords_for_group(self) -> list[tuple[list[set[int]] | None, float]]:
        potential_chords = []
        estimated_key = self.get_estimated_key()

        for estimate in self.source_key_estimates:
            chords_for_estimate = estimate.get_potential_chords(estimated_key)
            if len(chords_for_estimate[0]) == 0:
                raise ValueError("No potential chords found for estimate at time " + str(estimate.start_time) + " with estimate " + str(estimate) + " in the master set " + str(estimated_key))
            potential_chords.append(chords_for_estimate)

        return potential_chords
    
    def can_play_chord_at_time(self, chord: set[int], start_time: float, end_time: float) -> bool:
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
            return ([(p, 1) for p in all_chords], self.start_time) 

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
            return ([(p, 4) for p in perfect_chords_found], self.start_time)  # return only perfect chords found
        
        to_return = (list(zip(potential_chords, potential_chords_fitness)), self.start_time)  # return all potential chords found
        if len(to_return[0]) == 0:
            if not remove_echoing_notes:
                return self.get_potential_chords(master_set_id, remove_echoing_notes=True, two_note_chords=two_note_chords)  # try again removing echoing notes
            elif not two_note_chords:
                return self.get_potential_chords(master_set_id, remove_echoing_notes=remove_echoing_notes, two_note_chords=True)  # try again allowing two note chords
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