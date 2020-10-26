def get_rollout_length_from_schedule(schedule, epoch):
    return int(min(
        max(schedule[0] +
            (epoch - schedule[2])/(schedule[3] - schedule[2]) *
            (schedule[1] - schedule[0]),
            schedule[0]),
        schedule[1]))
