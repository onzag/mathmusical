import pretty_midi

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
    for note in quantized_combined_track_with_echo.notes:
        working_group = []
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

def extract_chords(quantized_combined_track_with_echo: pretty_midi.Instrument, key_estimates: list[KeyEstimateGrouped]) -> pretty_midi.Instrument:
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
    actual_start_time_for_next_added = None
        
    for chords_for_estimate in all_potential_chords:
        if len(intersection_so_far) == 0:
            if chords_for_estimate[0] is None:
                actual_start_time_for_next_added = chords_for_estimate[1]
            else:
                if actual_start_time_for_next_added is not None:
                    chords_for_estimate = (chords_for_estimate[0], actual_start_time_for_next_added)
                    actual_start_time_for_next_added = None
                intersection_so_far.append(chords_for_estimate)
        else:
            last_intersection = intersection_so_far[-1]
            # check if last_intersection contains any chord that is compatible with any chord in the current chords_for_estimate
            new_intersection = []
            for last_chord_possible, last_chord_fitness in last_intersection[0]:
                # check if this last chord is compatible with any chord in the current chords_for_estimate
                if chords_for_estimate[0] is None:
                    # all chords are possible because there are no given
                    new_intersection.append((last_chord_possible, last_chord_fitness))
                    continue
                # otherwise check for compatibility
                for current_chord_possible, current_chord_fitness in chords_for_estimate[0]:
                    # if we find a match, we add it to the new intersection
                    if last_chord_possible == current_chord_possible:
                        new_intersection.append((last_chord_possible, max(last_chord_fitness, current_chord_fitness)))
                        break

            if len(new_intersection) == 0:
                # no intersection found, we add a new entry
                if actual_start_time_for_next_added is not None:
                    chords_for_estimate = (chords_for_estimate[0], actual_start_time_for_next_added)
                    actual_start_time_for_next_added = None
                intersection_so_far.append(chords_for_estimate)
            else:
                last_intersection = (new_intersection, last_intersection[1])  # update the last intersection with the new intersection

    final_chords = []
    for chords_entry in intersection_so_far:
        # pick the chord with the highest fitness
        best_chord = max(chords_entry[0], key=lambda x: x[1])
        # add it to the final chords with its start time
        final_chords.append((best_chord[0], chords_entry[1]))

    for i in range(len(final_chords)):
        chord_notes, start_time = final_chords[i]
        next_chord_start_time = last_chord_end_time
        if i + 1 < len(final_chords):
            next_chord_start_time = final_chords[i + 1][1]

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
            potential_chords.append(chords_for_estimate)

        return potential_chords
    
    def __repr__(self):
        return f"KeyEstimateGrouped(start_time={self.start_time}, potential_master_set_ids={self.potential_master_set_ids}, estimated_key={self.get_estimated_key()}, notes_to_drop={self.get_notes_to_drop()})"

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
        return notes_to_drop
    
    def get_potential_chords(self, master_set_id: int) -> tuple[list[set[int]] | None, float]:
        notes_to_drop = self.get_notes_to_drop(master_set_id)
        reference_notes_without_drops = self.reference_notes_mod_12.difference(set(notes_to_drop))

        if len(reference_notes_without_drops) <= 1:
            # single notes are compatible with every chord basically
            # so it shall return None to indicate that
            return (None, self.start_time)  # no notes left to form chords
        
        potential_chords = []
        potential_chords_fitness = []
        perfect_chords_found = []
        # first try all chord shapes starting from each note in the reference notes
        for note in reference_notes_without_drops:
            # try each chord shape
            for shape_def in CHORD_SHAPES_PURE:
                # build the chord shape starting from this note
                chord_shape = set()
                invalid_shape = False

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
                if not invalid_shape:
                    potential_chords.append(chord_shape)

                    # check if the chord shape is a subset of our reference notes
                    if chord_shape.issubset(reference_notes_without_drops):
                        perfect_chords_found.append(chord_shape)

                    # check how many notes from the chord shape are in our reference notes
                    notes_in_shape_and_reference = chord_shape.intersection(reference_notes_without_drops)
                    fitness = len(notes_in_shape_and_reference)
                    potential_chords_fitness.append(fitness)

        if len(perfect_chords_found) > 0:
            return ([(p, 4) for p in perfect_chords_found], self.start_time)  # return only perfect chords found
        
        return (list(zip(potential_chords, potential_chords_fitness)), self.start_time)  # return all potential chords found

        
    def __repr__(self):
        return f"KeyEstimate(start_time={self.start_time}, reference_notes_mod_12={self.reference_notes_mod_12}, valid_master_sets_ids={self.valid_master_sets_ids}, potential_master_set_ids={self.potential_master_set_ids})"

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