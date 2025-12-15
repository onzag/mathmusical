import pretty_midi

def combine_tracks_while_avoiding_collisions(
    instuments: list[pretty_midi.Instrument],
    also_avoid: pretty_midi.Instrument = None,
    collision_threshold_before_octave_increase: int = None,
    octave_increase: int = 0,
) -> pretty_midi.Instrument:
    combined_track = pretty_midi.Instrument(program=0, is_drum=False)

    tracks_couldnt_be_added = []
    for track in instuments:
        notes_to_append = []
        above_threshold = False
        collision_count = 0
        for note in track.notes:
            # Check for collisions
            collision = False
            note_pitch = (note.pitch + (octave_increase * 12))
            for already_added_note in combined_track.notes:
                if note.start == already_added_note.start and note_pitch == already_added_note.pitch:
                    collision = True
                    collision_count += 1
                    break
            if also_avoid:
                for note2 in also_avoid.notes:
                    if note.start == note2.start and note_pitch == note2.pitch:
                        collision = True
                        collision_count += 1
                        break
            if not collision:
                notes_to_append.append(note)
            if collision_threshold_before_octave_increase is not None and collision_count >= collision_threshold_before_octave_increase:
                above_threshold = True
                break

        if above_threshold:
            tracks_couldnt_be_added.append(track)
            continue

        for note in notes_to_append:
            combined_track.notes.append(pretty_midi.Note(
                velocity=note.velocity,
                pitch=note.pitch + (octave_increase * 12),
                start=note.start,
                end=note.end,
            ))

    if len(tracks_couldnt_be_added) > 0:
        all_tracks_to_merge = [combined_track] + tracks_couldnt_be_added
        # try again with octave increase
        return combine_tracks_while_avoiding_collisions(
            all_tracks_to_merge,
            also_avoid,
            collision_threshold_before_octave_increase,
            octave_increase + 1,
        )

    return combined_track

def combine_tracks(instuments: list[pretty_midi.Instrument]) -> tuple[pretty_midi.Instrument, pretty_midi.Instrument, pretty_midi.Instrument, pretty_midi.Instrument, float]:
    general_tracks = [inst for inst in instuments if not inst.is_drum]

    drum_tracks = [inst for inst in instuments if inst.is_drum]
    drum_track = pretty_midi.Instrument(program=0, is_drum=True)
    for dt in drum_tracks:
        for note in dt.notes:
            drum_track.notes.append(note)

    harmonic_tracks = []
    melodic_tracks = []
    # we will check that they are on average playing one note at a time
    # however if they are too low we will consider them bass tracks
    bass_tracks = []

    average_note_duration_combined = 0.0
    notes_counted = 0
    for track in general_tracks:
        average_note_pictch_for_track = sum([note.pitch for note in track.notes]) / len(track.notes) if len(track.notes) > 0 else 0
        simulateous_notes = 0
        simulateous_notes_final_values = []
        current_time = 0.0
        time_window = 0.25  # 250 ms time window
        for note in track.notes:
            average_note_duration_combined += note.end - note.start
            notes_counted += 1
            
            if current_time - time_window <= note.start < current_time + time_window:
                simulateous_notes += 1
            else:
                current_time = note.start
                simulateous_notes_final_values.append(simulateous_notes)
                simulateous_notes = 1

        simulateous_notes_final_values.append(simulateous_notes)

        # calculate the average
        avg_simulateous = sum(simulateous_notes_final_values) / len(simulateous_notes_final_values) if len(simulateous_notes_final_values) > 0 else 0
        if avg_simulateous <= 1.8:
            if average_note_pictch_for_track < 48:  # MIDI pitch for C3 is 48
                bass_tracks.append(track)
            else:
                melodic_tracks.append(track)
        else:
            harmonic_tracks.append(track)

    combined_harmoic_tracks = combine_tracks_while_avoiding_collisions(harmonic_tracks + bass_tracks)
    melodic_track = combine_tracks_while_avoiding_collisions(melodic_tracks, combined_harmoic_tracks, 5)

    combined_harmoic_tracks.notes.sort(key=lambda n: n.start)
    melodic_track.notes.sort(key=lambda n: n.start)
    drum_track.notes.sort(key=lambda n: n.start)

    # rename tracks harmonic to be named left hand and melodic to be named right hand
    combined_harmoic_tracks.name = "Left Hand Track"
    melodic_track.name = "Right Hand Track"
    drum_track.name = "Drum Track"

    combined_track_with_echo = combine_tracks_while_avoiding_collisions([combined_harmoic_tracks, melodic_track])
    combined_track_with_echo.name = "Combined Track with Echo"
    combined_track_with_echo.notes.sort(key=lambda n: n.start)

    return (combined_harmoic_tracks, melodic_track, combined_track_with_echo, drum_track, average_note_duration_combined / notes_counted if notes_counted > 0 else 0.0)