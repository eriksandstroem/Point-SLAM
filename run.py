import argparse
from datetime import datetime

from src import config
from src.Point_SLAM import Point_SLAM
from src.common import setup_seed



def main():
    parser = argparse.ArgumentParser(
        description='Arguments for running the Point-SLAM.'
    )
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--input_folder', type=str,
                        help='input folder, this have higher priority, can overwrite the one in config file')
    parser.add_argument('--output', type=str,
                        help='output folder, this have higher priority, can overwrite the one in config file')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--no_wandb', action='store_true')

    def optional_int(string):
        return None if string == "None" else int(string)
    parser.add_argument("--stop", type=optional_int,
                        help="stop after n frames")

    args = parser.parse_args()

    cfg = config.load_config(args.config, 'configs/point_slam.yaml')

    setup_seed(cfg["setup_seed"])

    if args.stop:
        cfg["mapping"]["ckpt_freq"] = args.stop
        cfg["mapping"]["keyframe_every"] = 10


    now = datetime.now()
    time_string = now.strftime("%Y%m%d_%H%M%S") if args.stop is None else None

    slam = Point_SLAM(cfg, args, time_string=time_string)

    slam.run()


if __name__ == '__main__':
    main()
