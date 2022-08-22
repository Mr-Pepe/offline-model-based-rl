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
                if 0 < train_model_every <= steps_since_model_training:
                    if model_trained:
                        # Ignore init_steps if pretraining happened
                        return True

                    if step >= init_steps:
                        return True

    return False
