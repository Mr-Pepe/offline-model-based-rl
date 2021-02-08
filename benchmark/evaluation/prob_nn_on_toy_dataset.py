from benchmark.test.test_environment_model import \
    test_probabilistic_model_trains_on_toy_dataset
import argparse
from benchmark.utils.str2bool import str2bool

parser = argparse.ArgumentParser()
parser.add_argument('--augment_loss', type=str2bool, default=True)
parser.add_argument('--r_bounds_trainable', type=str2bool, default=False)
parser.add_argument('--steps_per_plot', type=int, default=100)
args = parser.parse_args()

test_probabilistic_model_trains_on_toy_dataset(steps=50000, plot=True,
                                               augment_loss=args.augment_loss,
                                               r_bounds_trainable=args.r_bounds_trainable,
                                               steps_per_plot=args.steps_per_plot)
