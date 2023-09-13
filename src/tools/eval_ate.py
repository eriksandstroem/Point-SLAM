import sys
import torch
import numpy
import os
import argparse
sys.path.append('.')
from src.common import get_tensor_from_camera
from src import config


def associate(first_list, second_list, offset=0.0, max_difference=0.02):
    """
    Associate two dictionaries of (stamp,data). As the time stamps never match exactly, we aim
    to find the closest match for every input tuple.

    Args:
        first_list -- first dictionary of (stamp,data) tuples
        second_list -- second dictionary of (stamp,data) tuples
        offset -- time offset between both dictionaries (e.g., to model the delay between the sensors)
        max_difference -- search radius for candidate generation

    Returns:
        matches -- list of matched tuples ((stamp1,data1),(stamp2,data2))

    """
    first_keys = list(first_list.keys())
    second_keys = list(second_list.keys())
    potential_matches = [(abs(a - (b + offset)), a, b)
                         for a in first_keys
                         for b in second_keys
                         if abs(a - (b + offset)) < max_difference]
    potential_matches.sort()
    matches = []
    for diff, a, b in potential_matches:
        if a in first_keys and b in second_keys:
            first_keys.remove(a)
            second_keys.remove(b)
            matches.append((a, b))

    matches.sort()
    return matches


def align(model, data):
    """Align two trajectories using the method of Horn (closed-form).

    Args:
        model -- first trajectory (3xn)
        data -- second trajectory (3xn)

    Returns:
        rot -- rotation matrix (3x3)
        trans -- translation vector (3x1)
        trans_error -- translational error per point (1xn)

    """
    numpy.set_printoptions(precision=3, suppress=True)
    model_zerocentered = model - model.mean(1)
    data_zerocentered = data - data.mean(1)

    W = numpy.zeros((3, 3))
    for column in range(model.shape[1]):
        W += numpy.outer(model_zerocentered[:,
                         column], data_zerocentered[:, column])
    U, d, Vh = numpy.linalg.linalg.svd(W.transpose())
    S = numpy.matrix(numpy.identity(3))
    if (numpy.linalg.det(U) * numpy.linalg.det(Vh) < 0):
        S[2, 2] = -1
    rot = U*S*Vh
    trans = data.mean(1) - rot * model.mean(1)

    model_aligned = rot * model + trans
    alignment_error = model_aligned - data

    trans_error = numpy.sqrt(numpy.sum(numpy.multiply(
        alignment_error, alignment_error), 0)).A[0]

    return rot, trans, trans_error


def plot_traj(ax, stamps, traj, style, color, label):
    """
    Plot a trajectory using matplotlib. 

    Args:
        ax -- the plot
        stamps -- time stamps (1xn)
        traj -- trajectory (3xn)
        style -- line style
        color -- line color
        label -- plot legend

    """
    stamps.sort()
    interval = numpy.median([s-t for s, t in zip(stamps[1:], stamps[:-1])])
    x = []
    y = []
    last = stamps[0]
    for i in range(len(stamps)):
        if stamps[i]-last < 2*interval:
            x.append(traj[i][0])
            y.append(traj[i][1])
        elif len(x) > 0:
            ax.plot(x, y, style, color=color, label=label)
            label = ""
            x = []
            y = []
        last = stamps[i]
    if len(x) > 0:
        ax.plot(x, y, style, color=color, label=label)


def evaluate_ate(first_list, second_list, plot="", scene="", use_alignment=False, _args=""):
    # parse command line
    parser = argparse.ArgumentParser(
        description='This script computes the absolute trajectory error from the ground truth trajectory and the estimated trajectory.')

    parser.add_argument(
        '--offset', help='time offset added to the timestamps of the second file (default: 0.0)', default=0.0)
    parser.add_argument(
        '--scale', help='scaling factor for the second trajectory (default: 1.0)', default=1.0)
    parser.add_argument(
        '--max_difference', help='maximally allowed time difference for matching entries (default: 0.02)', default=0.02)
    parser.add_argument(
        '--save', help='save aligned second trajectory to disk (format: stamp2 x2 y2 z2)')
    parser.add_argument('--save_associations',
                        help='save associated first and aligned second trajectory to disk (format: stamp1 x1 y1 z1 stamp2 x2 y2 z2)')
    parser.add_argument(
        '--plot', help='plot the first and the aligned second trajectory to an image (format: png)')
    parser.add_argument(
        '--verbose', help='print all evaluation data (otherwise, only the RMSE absolute translational error in meters after alignment will be printed)', action='store_true')
    args = parser.parse_args(_args)
    args.plot = plot

    matches = associate(first_list, second_list, float(
        args.offset), float(args.max_difference))
    if len(matches) < 2:
        raise ValueError(
            "Couldn't find matching timestamp pairs between groundtruth and estimated trajectory! \
            Did you choose the correct sequence?")

    first_xyz = numpy.matrix(
        [[float(value) for value in first_list[a][0:3]] for a, b in matches]).transpose()
    second_xyz = numpy.matrix([[float(value)*float(args.scale)
                              for value in second_list[b][0:3]] for a, b in matches]).transpose()

    if use_alignment:
        rot, trans, trans_error = align(second_xyz, first_xyz)
        second_xyz_aligned = rot * second_xyz + trans
    else:
        alignment_error = second_xyz - first_xyz
        trans_error = numpy.sqrt(numpy.sum(numpy.multiply(
            alignment_error, alignment_error), 0)).A[0]
        second_xyz_aligned = second_xyz

    first_stamps = list(first_list.keys())
    first_stamps.sort()
    first_xyz_full = numpy.matrix(
        [[float(value) for value in first_list[b][0:3]] for b in first_stamps]).transpose()

    second_stamps = list(second_list.keys())
    second_stamps.sort()
    second_xyz_full = numpy.matrix([[float(value)*float(args.scale)
                                   for value in second_list[b][0:3]] for b in second_stamps]).transpose()
    if use_alignment:
        second_xyz_full_aligned = rot * second_xyz_full + trans
    else:
        second_xyz_full_aligned = second_xyz_full

    if args.save_associations:
        file = open(args.save_associations, "w")
        file.write("\n".join(["%f %f %f %f %f %f %f %f" % (a, x1, y1, z1, b, x2, y2, z2) for (
            a, b), (x1, y1, z1), (x2, y2, z2) in zip(matches, first_xyz.transpose().A, second_xyz_aligned.transpose().A)]))
        file.close()

    if args.save:
        file = open(args.save, "w")
        file.write("\n".join(["%f " % stamp+" ".join(["%f" % d for d in line])
                   for stamp, line in zip(second_stamps, second_xyz_full_aligned.transpose().A)]))
        file.close()
    align_option = 'aligned' if use_alignment else 'no_align'

    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ATE = numpy.sqrt(
            numpy.dot(trans_error, trans_error) / len(trans_error))
        scene_name = scene.split('.')[0].split(
            '/')[-2]+'_'+scene.split('.')[0].split('/')[-1]
        ax.set_title(
            f'ate-rmse of {len(trans_error)} pose pairs ({align_option}):{float(ATE):0.4f}m {scene_name}')
        plot_traj(ax, first_stamps, first_xyz_full.transpose().A,
                  '-', "black", "ground truth")
        ax.plot(first_xyz_full.transpose().A[0][0], first_xyz_full.transpose(
        ).A[0][1], marker="o", markersize=5, markerfacecolor="green", label="start gt")
        ax.plot(first_xyz_full.transpose().A[-1][0], first_xyz_full.transpose(
        ).A[-1][1], marker="o", markersize=5, markerfacecolor="yellow", label="end gt")

        plot_traj(ax, second_stamps, second_xyz_full_aligned.transpose(
        ).A, '-', "blue", "estimated")
        ax.plot(second_xyz_full_aligned.transpose().A[0][0], second_xyz_full_aligned.transpose(
        ).A[0][1], marker="*", markersize=5, markerfacecolor="cyan", label="start estimated")
        ax.plot(second_xyz_full_aligned.transpose().A[-1][0], second_xyz_full_aligned.transpose(
        ).A[-1][1], marker="*", markersize=5, markerfacecolor="purple", label="end estimated")

        label = "difference"
        for (a, b), (x1, y1, z1), (x2, y2, z2) in zip(matches, first_xyz.transpose().A, second_xyz_aligned.transpose().A):
            # ax.plot([x1,x2],[y1,y2],'-',color="red",label=label)
            label = ""
        ax.legend()
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        plt.savefig(args.plot, dpi=300)

    return {
        "compared_pose_pairs": (len(trans_error)),
        "absolute_translational_error.rmse": numpy.sqrt(numpy.dot(trans_error, trans_error) / len(trans_error)),
        "absolute_translational_error.mean": numpy.mean(trans_error),
        "absolute_translational_error.median": numpy.median(trans_error),
        "absolute_translational_error.std": numpy.std(trans_error),
        "absolute_translational_error.min": numpy.min(trans_error),
        "absolute_translational_error.max": numpy.max(trans_error),
    }


def evaluate(poses_gt, poses_est, plot, scene, use_alignment=False):

    poses_gt = poses_gt.cpu().numpy()
    poses_est = poses_est.cpu().numpy()

    N = poses_gt.shape[0]
    poses_gt = dict([(i, poses_gt[i]) for i in range(N)])
    poses_est = dict([(i, poses_est[i]) for i in range(N)])

    results = evaluate_ate(poses_gt, poses_est, plot, scene, use_alignment)
    print(results)
    return results


def convert_poses(c2w_list, N, gt=True):
    poses = []
    mask = torch.ones(N+1).bool()
    for idx in range(0, N+1):
        if gt:
            # some frame have `nan` or `inf` in gt pose of ScanNet,
            # but our system have estimated camera pose for all frames
            # therefore, when calculating the pose error, we need to mask out invalid pose
            if torch.isinf(c2w_list[idx]).any():
                mask[idx] = 0
                continue
            if torch.isnan(c2w_list[idx]).any():
                mask[idx] = 0
                continue
        poses.append(get_tensor_from_camera(c2w_list[idx], Tquad=True))
    poses = torch.stack(poses)
    return poses, mask


if __name__ == '__main__':
    """
    This ATE evaluation code is modified upon the evaluation code in lie-torch.
    """

    parser = argparse.ArgumentParser(
        description='Arguments to eval the tracking ATE.'
    )
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--output', type=str,
                        help='output folder, this have higher priority, can overwrite the one inconfig file')
    parser.add_argument('--no_align', default=False, action='store_true',
                        help='if to align the first and second trajectory before evaluating')

    args = parser.parse_args()
    cfg = config.load_config(args.config, 'configs/point_slam.yaml')
    output = cfg['data']['output'] if args.output is None else args.output
    ckptsdir = f'{output}/ckpts'
    if os.path.exists(ckptsdir):
        ckpts = [os.path.join(ckptsdir, f)
                 for f in sorted(os.listdir(ckptsdir)) if 'tar' in f]
        if len(ckpts) > 0:
            ckpt_path = ckpts[-1]
            ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
            estimate_c2w_list = ckpt['estimate_c2w_list']
            gt_c2w_list = ckpt['gt_c2w_list']
            N = ckpt['idx']

            poses_gt, mask = convert_poses(gt_c2w_list, N)
            poses_est, _ = convert_poses(estimate_c2w_list, N)
            poses_est = poses_est[mask]
            align_option = 'aligned' if not args.no_align else 'no_align'
            results = evaluate(poses_gt, poses_est,
                               plot=f'{output}/eval_ate_{align_option}.png',
                               scene=args.config, use_alignment=not args.no_align)
