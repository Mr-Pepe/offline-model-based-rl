# Based on https://spinningup.openai.com

import os
import os.path as osp
import time

import gym
import joblib
import torch

from offline_mbrl.utils.logx import EpochLogger
from offline_mbrl.utils.setup_test_env import setup_test_env
from offline_mbrl.utils.str2bool import str2bool


def load_policy_and_env(fpath, itr="last", deterministic=False, test_env=True):
    """
    Load a policy from save, whether it's TF or PyTorch, along with RL env.

    Not exceptionally future-proof, but it will suffice for basic uses of the
    Spinning Up implementations.

    Checks to see if there's a tf1_save folder. If yes, assumes the model
    is tensorflow and loads it that way. Otherwise, loads as if there's a
    PyTorch save.
    """

    # handle which epoch to load from
    if itr == "last":
        # check filenames for epoch (AKA iteration) numbers, find maximum value

        pytsave_path = osp.join(fpath, "pyt_save")
        # Each file in this folder has naming convention 'modelXX.pt', where
        # 'XX' is either an integer or empty string. Empty string case
        # corresponds to len(x)==8, hence that case is excluded.
        saves = [
            int(x.split(".")[0][5:])
            for x in os.listdir(pytsave_path)
            if len(x) > 8 and "model" in x
        ]

        itr = str(max(saves)) if len(saves) > 0 else ""

    else:
        assert isinstance(
            itr, int
        ), f"Bad value provided for {itr} (needs to be int or 'last')."

    # load the get_action function
    get_action = load_pytorch_policy(fpath, itr, deterministic)

    # try to load environment from save
    # (sometimes this will fail because the environment could not be pickled)
    try:
        state = joblib.load(osp.join(fpath, "vars" + itr + ".pkl"))
        env = state["env"]

        if test_env:
            env = gym.make(env.spec.id)

    except Exception:  # pylint: disable=broad-except
        env = None

    return env, get_action


def load_pytorch_policy(fpath, itr):
    """Load a pytorch policy saved with Spinning Up Logger."""

    fname = osp.join(fpath, "pyt_save", "agent" + itr + ".pt")
    print(f"\n\nLoading from {fname}.\n\n")

    model = torch.load(fname, map_location="cpu")

    # make function for producing an action given a single state
    def get_action(x):
        with torch.no_grad():
            x = torch.as_tensor(x, dtype=torch.float32)
            action = model.act(x, True)
        return action

    return get_action


def run_policy(env, get_action, max_ep_len=None, num_episodes=100, render=True):

    assert (
        env is not None
    ), """Environment not found!\n\n It looks like the environment wasn't saved,
        and we can't run the agent in it. :( \n\n Check out the readthedocs
        page on Experiment Outputs for how to handle this situation."""

    logger = EpochLogger()
    r = 0
    d = False
    ep_ret = 0
    ep_len = 0
    n = 0
    o = setup_test_env(env)
    while n < num_episodes:
        if render:
            env.render()
            time.sleep(1e-3)

        a = get_action(o).cpu().numpy()
        o, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            print(f"Episode {n} \t EpRet {ep_ret:.3f} \t EpLen {ep_len}")
            r = 0
            d = False
            ep_ret = 0
            ep_len = 0
            o = setup_test_env(env)
            n += 1

    logger.log_tabular("EpRet", with_min_and_max=True)
    logger.log_tabular("EpLen", average_only=True)
    logger.dump_tabular()


def test_agent(
    test_env,
    agent,
    max_ep_len,
    num_test_episodes,
    logger,
    render=False,
    buffer=None,
    use_setup=False,
):
    sum_ep_ret = 0
    for _ in range(num_test_episodes):
        d = False
        ep_ret = 0
        ep_len = 0
        if use_setup:
            o = setup_test_env(test_env)
        else:
            o = test_env.reset()

        while not (d or (ep_len == max_ep_len)):
            # Take deterministic actions at test time
            a = agent.act(o, True).cpu().numpy()
            o2, r, d, _ = test_env.step(a)
            if render:
                test_env.render()

            if buffer is not None:
                buffer.store(
                    torch.as_tensor(o),
                    torch.as_tensor(a),
                    torch.as_tensor(r),
                    torch.as_tensor(o2),
                    torch.as_tensor(d),
                )

            o = o2

            ep_ret += r
            ep_len += 1
        logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)
        sum_ep_ret += ep_ret

        render = False

    return sum_ep_ret / num_test_episodes


def main(args):
    env, get_action = load_policy_and_env(
        args.fpath,
        args.itr if args.itr >= 0 else "last",
        args.deterministic,
        args.test_env,
    )
    run_policy(env, get_action, args.len, args.episodes, not args.norender)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--fpath", type=str, default="")
    parser.add_argument("--len", "-l", type=int, default=0)
    parser.add_argument("--episodes", "-n", type=int, default=100)
    parser.add_argument("--norender", "-nr", action="store_true")
    parser.add_argument("--itr", "-i", type=int, default=-1)
    parser.add_argument("--deterministic", "-d", action="store_true")
    parser.add_argument("--test_env", type=str2bool, default=True)
    main(parser.parse_args())
