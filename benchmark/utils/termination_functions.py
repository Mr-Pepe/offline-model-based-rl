import torch


def hopper_termination_fn(next_obs):

    height = next_obs[:, 0]
    angle = next_obs[:, 1]
    not_done = torch.isfinite(next_obs).all(axis=-1) \
        * (next_obs[:, 1:] < 100).all(axis=-1) \
        * (height > .7) \
        * (torch.abs(angle) < .2)

    done = ~not_done
    done = done[:, None]
    return done


def half_cheetah_termination_fn(obs):

    done = torch.as_tensor([False]).repeat(len(obs))
    done = done[:, None]
    return done


def walker2d_termination_fn(next_obs):

    height = next_obs[:, 0]
    angle = next_obs[:, 1]
    not_done = (height > 0.8) \
        * (height < 2.0) \
        * (angle > -1.0) \
        * (angle < 1.0)
    done = ~not_done
    done = done[:, None]
    return done


def antmaze_termination_fn(next_obs):

    # This is technically not correct, as the environment terminates a
    # rollout when reaching the goal
    done = torch.as_tensor([False]).repeat(len(next_obs))
    done = done[:, None]
    return done


termination_functions = {
    'hopper': hopper_termination_fn,
    'half_cheetah': half_cheetah_termination_fn,
    'walker2d': walker2d_termination_fn,
    'antmaze': antmaze_termination_fn,
}

function_to_names_mapping = {
    'hopper': ['Hopper-v2',
               'hopper-random-v0',
               'hopper-medium-v0',
               'hopper-expert-v0',
               'hopper-medium-replay-v0',
               'hopper-medium-expert-v0'],
    'half_cheetah': ['HalfCheetah-v2',
                     'halfcheetah-random-v0',
                     'halfcheetah-medium-v0',
                     'halfcheetah-expert-v0',
                     'halfcheetah-medium-replay-v0',
                     'halfcheetah-medium-expert-v0'],
    'walker2d': ['Walker2d-v2',
                 'walker2d-random-v0',
                 'walker2d-medium-v0',
                 'walker2d-expert-v0',
                 'walker2d-medium-replay-v0',
                 'walker2d-medium-expert-v0',
                 ],
    'antmaze': ['antmaze-umaze-v0',
                'antmaze-umaze-diverse-v0',
                'antmaze-medium-diverse-v0',
                'antmaze-medium-play-v0',
                'antmaze-large-diverse-v0',
                'antmaze-large-play-v0',
                ],
}


def get_termination_function(env_name):
    for fn_name in function_to_names_mapping:
        if env_name in function_to_names_mapping[fn_name]:
            return termination_functions[fn_name]

    return None
