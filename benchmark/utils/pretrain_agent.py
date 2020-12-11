from benchmark.utils.mazes import plot_antmaze_umaze, plot_maze2d_umaze
from benchmark.utils.virtual_rollouts import generate_virtual_rollouts
from benchmark.utils.replay_buffer import ReplayBuffer
import matplotlib.pyplot as plt


def pretrain_agent(agent,
                   model,
                   real_buffer,
                   n_steps=1000,
                   n_random_actions=0,
                   max_rollout_length=1000,
                   pessimism=0,
                   exploration_mode='state',
                   n_rollouts=1,
                   debug=False):

    virtual_buffer = ReplayBuffer(model.obs_dim,
                                  model.act_dim,
                                  int(1e6),
                                  device=next(model.parameters()).device)

    prev_obs = None
    f = plt.figure()

    print('')

    for step in range(n_steps):
        print("Pretrain agent: Step {}/{}".format(step+1, n_steps), end='\r')

        rollouts, prev_obs = generate_virtual_rollouts(
            model,
            agent,
            real_buffer,
            steps=1,
            n_rollouts=n_rollouts,
            stop_on_terminal=True,
            pessimism=pessimism,
            random_action=step < n_random_actions,
            prev_obs=prev_obs,
            max_rollout_length=max_rollout_length,
            exploration_mode=exploration_mode
        )

        virtual_buffer.store_batch(rollouts['obs'],
                                   rollouts['act'],
                                   rollouts['rew'],
                                   rollouts['next_obs'],
                                   rollouts['done'])

        if step > 2000:
            agent.multi_update(2, virtual_buffer)

        if debug and (step + 1) % 500 == 0:
            f.clear()

            plot_antmaze_umaze(buffer=virtual_buffer)

            plt.draw()
            plt.pause(0.001)
