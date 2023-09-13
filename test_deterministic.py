import torch
import argparse
import subprocess
import os
import sys

config_folder = "configs"
outputs_folder = "output"

configs = {"Replica": "Replica/office0.yaml",
    # "ScanNet": "ScanNet/scene0103.yaml",
    #"TUM_RGBD": "TUM_RGBD/freiburg1_desk.yaml"
}

outputs = {"Replica": "Replica/office0/ckpts",
    # "ScanNet": "ScanNet/scene0103/reference/ckpts",
    # "TUM_RGBD": "TUM_RGBD/freiburg1_desk/reference/ckpts"
}

tensor_names = ["geo_feats", "col_feats", "gt_c2w_list", "estimate_c2w_list"]


def run(args):
    for k, v in configs.items():
        print("running {}".format(k))
        proc = subprocess.Popen(["python", "run.py", os.path.join(
            config_folder, v), "--stop={}".format(args.n_frames)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for c in iter(lambda: proc.stdout.read(1), b""):
            sys.stdout.buffer.write(c)
        print("{} terminated".format(k))


def rename(args, name):
    cwd = os.getcwd()
    for k, v in outputs.items():
        folder = os.path.join(cwd, outputs_folder, v)
        os.chdir(folder)
        os.rename("{:05d}.tar".format(args.n_frames), "{}.tar".format(name))
        os.chdir(cwd)


def compare():
    cwd = os.getcwd()
    for k, v in outputs.items():
        folder = os.path.join(cwd, outputs_folder, v)
        os.chdir(folder)
        ref = torch.load("ref.tar")
        new = torch.load("new.tar")
        for tensor_name in tensor_names:
            print(k, tensor_name, torch.equal(
                ref[tensor_name], new[tensor_name]))


def main(args):
    import time
    tic = time.perf_counter()
    run(args)
    if args.gen_ref:
        rename(args, "ref")
    else:
        rename(args, "new")
        compare()
    toc = time.perf_counter()
    print("elapsed time", toc-tic)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_frames", type=int, default=50,
                        help="number of frames to run the scripts")
    parser.add_argument("--gen_ref", action="store_true")
    parser.set_defaults(gen_ref=False)
    args = parser.parse_args()

    main(args)
