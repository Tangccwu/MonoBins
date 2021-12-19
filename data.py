import argparse
import os
from posixpath import split

from numpy.core.numeric import False_
file_dir = os.path.dirname(__file__)  # the directory that options.py resides in
# print(file_dir) c:/Users/20811/Desktop/modify 当前目录


# parser.add_argument("--gt_path", default='../dataset/kitti/sync/', type=str,
#                     help="path to dataset")
# parser.add_argument('--filenames_file',
#                     default="./train_test_inputs/kitti_eigen_train_files_with_gt.txt",
#                     type=str, help='path to the filenames text file')

from dataloader import SeqDataLoader
from dataloaderT import Monodataset,Seq2DataLoader
from PIL import Image
def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Training script. Default values of all arguments are recommended for reproducibility', fromfile_prefix_chars='@',
                                     conflict_handler='resolve')
    parser.add_argument("--dataset", default='kitti', type=str, help="Dataset to train on")
    parser.add_argument("--data_path", default='E://kitti//sync', type=str,
                        help="path to dataset")
    parser.add_argument('--data_path_eval',
                        default="E:/kitti/sync/",
                        type=str, help='path to the data for online evaluation')
    parser.add_argument('--gt_path_eval', default="E:/kitti/sync/test/",
                        type=str, help='path to the groundtruth data for online evaluation')
    parser.add_argument("--input_height",type=int,help="input image height",default=352)
    parser.add_argument("--input_width",type=int,help="input image width",default=704)
    parser.add_argument("--split",type=str,
                                 help="which training split to use",
                                 choices=["eigen_zhou", "eigen_full", "odom", "benchmark"],
                                 default="eigen_zhou")
    parser.add_argument("--frame_ids",
                                 nargs="+",# 表示参数可以设置一个或者多个
                                 type=int,
                                 help="frames to load",
                                 default=[0, -1, 1])
    
    parser.add_argument('--filenames_file',
                            # default="./train_test_inputs/kitti_eigen_train_files_with_gt.txt",
                            default="./splits/eigen_zhou/train_files.txt",
                            # default=os.path.join(os.path.dirname(__file__), "splits","eigen_zhou", "{}_files.txt"),
                            type=str, help='path to the filenames text file')
    parser.add_argument('--filenames_file_eval',
                        # default="./train_test_inputs/kitti_eigen_train_files_with_gt.txt",
                        default="./splits/eigen_zhou/val_files.txt",
                        # default=os.path.join(os.path.dirname(__file__), "splits","eigen_zhou", "{}_files.txt"),
                        type=str, help='path to the filenames text file')                      
    parser.add_argument("--use_right",default=False)
    parser.add_argument("--do_kb_crop",default=True, action='store_true')
    parser.add_argument("--do_color_aug",default=False, action='store_true')
    parser.add_argument("--do_random_rotate",default=False, action='store_true')
    parser.add_argument("--png",
                                 default=True,
                                 help="if set, trains from raw KITTI png files (instead of jpgs)",
                                 action="store_true")
    parser.add_argument('--bs', default=16, type=int, help='batch size')
    parser.add_argument("--workers", default=11, type=int, help="Number of workers for data loading")
    args = parser.parse_args()
    args.batch_size = args.bs
    args.num_threads = args.workers
    

    train_loader = Seq2DataLoader(args, 'train').data
    test_loader = Seq2DataLoader(args, 'online_eval').data

    

