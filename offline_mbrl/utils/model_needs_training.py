def model_needs_training(
    step,
    use_model,
    buffer_size,
    init_steps,
    steps_since_model_training,
    train_model_every,
    model_trained,
):
    if use_model:
        if buffer_size > 0:
            if step < 0:
                # Train only once before offline training
                if not model_trained:
                    return True
            else:
                if (
                    steps_since_model_training >= train_model_every
                    and train_model_every > 0
                ):
                    if model_trained:
                        # Ignore init_steps if pretraining happened
                        return True
                    elif step >= init_steps:
                        return True

    return False
