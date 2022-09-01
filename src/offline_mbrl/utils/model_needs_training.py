from offline_mbrl.models.environment_model import EnvironmentModel


def model_needs_training(
    model: EnvironmentModel,
    step: int,
    buffer_size: int,
    init_steps: int,
    steps_since_model_training: int,
    train_model_every: int,
) -> bool:
    if model is None or buffer_size <= 0:
        return False

    if step < 0 and not model.has_been_trained_at_least_once:
        # Train once before offline training
        return True

    schedule_is_due = steps_since_model_training >= train_model_every > 0

    if schedule_is_due and step >= init_steps:
        return True

    if schedule_is_due and model.has_been_trained_at_least_once:
        # Ignore init steps if offline training happened
        return True

    return False
