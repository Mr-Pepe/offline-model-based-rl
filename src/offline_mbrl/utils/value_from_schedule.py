def get_value_from_schedule(schedule, epoch, is_float=False):
    value = min(
        max(
            schedule[0]
            + (epoch - schedule[2])
            / (schedule[3] - schedule[2])
            * (schedule[1] - schedule[0]),
            schedule[0],
        ),
        schedule[1],
    )
    if is_float:
        return value

    return int(value)
