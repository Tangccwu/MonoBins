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
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.distributed
from dataloader import SeqDataLoader
from dataloaderT import Monodataset,Seq2DataLoader
from loss import SILogLoss, BinsChamferLoss
from utils import RunningAverage, colorize
import numpy as np
from datetime import datetime as dt
from PIL import Image
import uuid
import wandb
from tqdm import tqdm
import model_io
import models
import networks
import utils
from layers import *
logging = True
PROJECT = "MDE-AdaBins"
def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines
def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield str(arg)
def is_rank_zero(args):
    return args.rank == 0
def predict_pose(args,inputs,model):
    outputs = {}
    if args.num_pose_frames == 2:
        pose_feats = {f_i: inputs["color", f_i] for f_i in args.frame_ids}
        for f_i in args.frame_ids[1:]:
            if f_i != "s":  
                pose_inputs = [pose_feats[f_i],pose_feats[0]]
            else: 
                pose_inputs = [pose_feats[0],pose_feats[f_i]]
            if args.pose_model_type == "separate_resnet":
                pose_inputs = [model["pose_encoder"](torch.cat(pose_inputs, 1))]
            elif args.pose_model_type == "posecnn":
                pose_inputs = torch.cat(pose_inputs, 1)
            axisangle, translation = model["pose"](pose_inputs)
            outputs[("axisangle", 0, f_i)] = axisangle
            outputs[("translation", 0, f_i)] = translation

            # Invert the matrix if the frame id is negative
            outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                axisangle[:, 0], translation[:, 0], invert=(f_i < 0))
    else:
        if args.pose_model_type in ["separate_resnet", "posecnn"]:
            pose_inputs = torch.cat(
                [inputs[("color", i)] for i in args.frame_ids if i != "s"], 1)

            if args.pose_model_type == "separate_resnet":
                pose_inputs = [model["pose_encoder"](pose_inputs)]

        axisangle, translation = model["pose"](pose_inputs)

        for i, f_i in enumerate(args.frame_ids[1:]):
            if f_i != "s":
                outputs[("axisangle", 0, f_i)] = axisangle
                outputs[("translation", 0, f_i)] = translation
                outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                    axisangle[:, i], translation[:, i])

    return outputs
def compute_reprojection_loss(args,pred,target):
    """Computes reprojection loss between a batch of predicted and target images
    """
    abs_diff = torch.abs(target - pred)
    l1_loss = abs_diff.mean(1, True)

    if args.no_ssim:
        reprojection_loss = l1_loss
    else:
        ssim_loss = args.ssim(pred, target).mean(1, True)
        reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

    return reprojection_loss
def compute_unsupervied_loss(args,inputs,outputs):
    losses = {}
    total_loss = 0
    loss = 0
    reprojection_losses = []
    target = inputs[("color", 0)]
    for frame_id in args.frame_ids[1:]:
        pred = outputs[("color", frame_id)]
        reprojection_losses.append(compute_reprojection_loss(pred, target))
    reprojection_losses = torch.cat(reprojection_losses, 1)

    if not args.disable_automasking:
        identity_reprojection_losses = []
        for frame_id in args.frame_ids[1:]:
            pred = inputs[("color", frame_id)]
            identity_reprojection_losses.append(compute_reprojection_loss(pred, target))
        identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

        if args.avg_reprojection:
            identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
        else:
            # save both images, and do min all at once below
            identity_reprojection_loss = identity_reprojection_losses
        identity_reprojection_loss = identity_reprojection_losses
    
    if args.avg_reprojection:
        reprojection_loss = reprojection_losses.mean(1, keepdim=True)
    else:
        reprojection_loss = reprojection_losses
    if not args.disable_automasking:
        # add random numbers to break ties
        identity_reprojection_loss += torch.randn(
            identity_reprojection_loss.shape).cuda() * 0.00001

        combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
    else:
        combined = reprojection_loss
    if combined.shape[1] == 1:
        to_optimise = combined
    else:
        to_optimise, idxs = torch.min(combined, dim=1)

    if not args.disable_automasking:
        outputs["identity_selection/{}".format(0)] = (
            idxs > identity_reprojection_loss.shape[1] - 1).float()
    loss += to_optimise.mean()


    return loss
def validate(args, model, test_loader, criterion_ueff, epoch, epochs, device='cpu'):
    with torch.no_grad():
        val_si = RunningAverage()
        # val_bins = RunningAverage()
        metrics = utils.RunningAverageDict()
        for batch in tqdm(test_loader, desc=f"Epoch: {epoch + 1}/{epochs}. Loop: Validation") if is_rank_zero(
                args) else test_loader:
            img = batch[('color',0)].cuda(device)
            depth = batch['depth_gt'].cuda(device)
            if 'has_valid_depth' in batch:
                if not batch['has_valid_depth']:
                    continue
            depth = depth.squeeze().unsqueeze(0).unsqueeze(0)
            bins, pred = model['UnetAdaptiveBins'](img)

            mask = depth > args.min_depth
            l_dense = criterion_ueff(pred, depth, mask=mask.to(torch.bool), interpolate=True)
            val_si.append(l_dense.item())

            pred = nn.functional.interpolate(pred, depth.shape[-2:], mode='bilinear', align_corners=True)

            pred = pred.squeeze().cpu().numpy()
            pred[pred < args.min_depth_eval] = args.min_depth_eval
            pred[pred > args.max_depth_eval] = args.max_depth_eval
            pred[np.isinf(pred)] = args.max_depth_eval
            pred[np.isnan(pred)] = args.min_depth_eval

            gt_depth = depth.squeeze().cpu().numpy()
            valid_mask = np.logical_and(gt_depth > args.min_depth_eval, gt_depth < args.max_depth_eval)
            if args.garg_crop or args.eigen_crop:
                gt_height, gt_width = gt_depth.shape
                eval_mask = np.zeros(valid_mask.shape)

                if args.garg_crop:
                    eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height),int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1

                elif args.eigen_crop:
                    if args.dataset == 'kitti':
                        eval_mask[int(0.3324324 * gt_height):int(0.91351351 * gt_height),int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1
                    else:
                        eval_mask[45:471, 41:601] = 1
            valid_mask = np.logical_and(valid_mask, eval_mask)
            metrics.update(utils.compute_errors(gt_depth[valid_mask], pred[valid_mask]))

        return metrics.get_value(), val_si
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Training script. Default values of all arguments are recommended for reproducibility', fromfile_prefix_chars='@',
                                     conflict_handler='resolve')
    parser.convert_arg_line_to_args = convert_arg_line_to_args
    parser.add_argument('--epochs', default=25, type=int, help='number of total epochs to run')
    parser.add_argument('--n-bins', '--n_bins', default=80, type=int,
                        help='number of bins/buckets to divide depth range into')
    parser.add_argument('--lr', '--learning-rate', default=0.000357, type=float, help='max learning rate')
    parser.add_argument('--plr', '--poselearning-rate', default=1e-4, type=float, help='max learning rate')
    parser.add_argument('--wd', '--weight-decay', default=0.1, type=float, help='weight decay')
    parser.add_argument('--w_chamfer', '--w-chamfer', default=0.1, type=float, help="weight value for chamfer loss")
    parser.add_argument('--div-factor', '--div_factor', default=25, type=float, help="Initial div factor for lr")
    parser.add_argument('--final-div-factor', '--final_div_factor', default=100, type=float,
                        help="final div factor for lr")

    parser.add_argument('--bs', default=1, type=int, help='batch size')
    parser.add_argument('--validate-every', '--validate_every', default=100, type=int, help='validation period')
    parser.add_argument('--gpu', default=None, type=int, help='Which gpu to use')
    parser.add_argument("--name", default="UnetAdaptiveBins")
    parser.add_argument("--norm", default="linear", type=str, help="Type of norm/competition for bin-widths",
                        choices=['linear', 'softmax', 'sigmoid'])
    parser.add_argument("--same-lr", '--same_lr', default=False, action="store_true",
                        help="Use same LR for all param groups")
    parser.add_argument("--distributed", default=False, action="store_true", help="Use DDP if set")
    parser.add_argument("--root", default=".", type=str,
                        help="Root folder to save data in")
    parser.add_argument("--resume", default='', type=str, help="Resume from checkpoint")

    parser.add_argument("--notes", default='', type=str, help="Wandb notes")
    parser.add_argument("--tags", default='sweep', type=str, help="Wandb tags")

    parser.add_argument("--workers", default=6, type=int, help="Number of workers for data loading")
    
    parser.add_argument("--dataset", default='kitti', type=str, help="Dataset to train on")
    parser.add_argument("--data_path", default='E:/kitti/sync/', type=str,
                        help="path to dataset")
    # parser.add_argument("--gt_path", default='E:/kitti/sync/train', type=str,
    #                     help="path to dataset")
    # parser.add_argument('--gt_path_eval', default="E:/kitti/sync/val/",
    #                     type=str, help='path to the groundtruth data for online evaluation')                    
    parser.add_argument('--filenames_file',default="./splits/eigen_zhou/train_files.txt",
                        # default="./train_test_inputs/kitti_eigen_train_files_with_gt.txt",
                        type=str, help='path to the filenames text file')
    parser.add_argument('--data_path_eval',
                        default="E:/kitti/sync/",
                        type=str, help='path to the data for online evaluation')
    parser.add_argument('--filenames_file_eval',default="./splits/eigen_zhou/val_files.txt",
                        # default="./train_test_inputs/kitti_eigen_test_files_with_gt.txt",
                        type=str, help='path to the filenames text file for online evaluation')

    # parser.add_argument("--dataset", default='nyu', type=str, help="Dataset to train on")
    # parser.add_argument("--data_path", default='../dataset/nyu/sync/', type=str,
    #                     help="path to dataset")
    # parser.add_argument("--gt_path", default='../dataset/nyu/sync/', type=str,
    #                     help="path to dataset")
    # parser.add_argument('--filenames_file',
    #                     default="./train_test_inputs/nyudepthv2_train_files_with_gt.txt",
    #                     type=str, help='path to the filenames text file')
                                           

    parser.add_argument('--input_height', type=int, help='input height', default=352)
    parser.add_argument('--input_width', type=int, help='input width', default=704)
    parser.add_argument('--max_depth', type=float, help='maximum depth in estimation', default=10)
    parser.add_argument('--min_depth', type=float, help='minimum depth in estimation', default=1e-3)

    # parser.add_argument('--do_random_rotate', default=False,
    #                     help='if set, will perform random rotation for augmentation',
    #                     action='store_true')
    parser.add_argument('--degree', type=float, help='random rotation maximum degree', default=2.5)
    parser.add_argument('--do_kb_crop',default=False, help='if set, crop input images as kitti benchmark images', action='store_true')
    parser.add_argument('--use_right', help='if set, will randomly use right images when train on KITTI',
                        action='store_true')

    # parser.add_argument('--data_path_eval',
    #                     default="../dataset/nyu/official_splits/test/",
    #                     type=str, help='path to the data for online evaluation')
    # parser.add_argument('--gt_path_eval', default="../dataset/nyu/official_splits/test/",
    #                     type=str, help='path to the groundtruth data for online evaluation')
    # parser.add_argument('--filenames_file_eval',
    #                     default="./train_test_inputs/nyudepthv2_test_files_with_gt.txt",
    #                     type=str, help='path to the filenames text file for online evaluation')


    # parser.add_argument('--data_path_eval',
    #                     default="E:/dataset/nyu/official_splits/test/",
    #                     type=str, help='path to the data for online evaluation')
    # parser.add_argument('--gt_path_eval', default="../dataset/nyu/official_splits/test/",
    #                     type=str, help='path to the groundtruth data for online evaluation')
    # parser.add_argument('--filenames_file_eval',
    #                     default="./train_test_inputs/nyudepthv2_test_files_with_gt.txt",
    #                     type=str, help='path to the filenames text file for online evaluation')

    parser.add_argument('--min_depth_eval', type=float, help='minimum depth for evaluation', default=1e-3)
    parser.add_argument('--max_depth_eval', type=float, help='maximum depth for evaluation', default=10)
    parser.add_argument('--eigen_crop', default=True, help='if set, crops according to Eigen NIPS14',
                        action='store_true')
    parser.add_argument('--garg_crop', help='if set, crops according to Garg  ECCV16', action='store_true')
    #######################################################################################################
    # new argement for Unsupervised learning
    parser.add_argument("--split",type=str,help="which training split to use",
                                  choices=["eigen_zhou", "eigen_full", "odom", "benchmark"],default="eigen_zhou")
    parser.add_argument("--frame_ids",nargs="+",type=int,help="frames to load",
                                 default=[0, -1, 1])
    parser.add_argument("--do_color_aug",default=False, action='store_true')
    parser.add_argument("--do_random_rotate",default=False, action='store_true')
    parser.add_argument("--png",default=True,
                                help="if set, trains from raw KITTI png files (instead of jpgs)",action="store_true")
    parser.add_argument("--pose_model_type",type=str,help="normal or shared",default="separate_resnet",
                                            choices=["posecnn", "separate_resnet"])
    parser.add_argument("--num_layers",type=int,help="number of resnet layers",default=18,
                                       choices=[18, 34, 50, 101, 152])
    parser.add_argument("--weights_init",type=str,help="pretrained or scratch",default="pretrained",
                                         choices=["pretrained", "scratch"])
    parser.add_argument("--pose_model_input",type=str,help="how many images the pose network gets",
                                             default="pairs",choices=["pairs", "all"])
    parser.add_argument("--disable_automasking",default=True,help="if set, trains from raw KITTI png files (instead of jpgs)",action="store_true")
    parser.add_argument("--no_ssim",default=False)
    parser.add_argument("--avg_reprojection", help="if set, uses average reprojection loss", action="store_true")
    args = parser.parse_args()
    args.batch_size = args.bs
    args.num_threads = args.workers
    args.mode = 'train'
    args.chamfer = args.w_chamfer > 0
    args.num_input_frames = len(args.frame_ids)
    args.num_pose_frames = 2 if args.pose_model_input == "pairs" else args.num_input_frames
    args.multigpu = False

    if args.root != "." and not os.path.isdir(args.root):
        os.makedirs(args.root)

    try:
        node_str = os.environ['SLURM_JOB_NODELIST'].replace('[', '').replace(']', '')
        nodes = node_str.split(',')

        args.world_size = len(nodes)
        args.rank = int(os.environ['SLURM_PROCID'])

    except KeyError as e:
        # We are NOT using SLURM
        args.world_size = 1
        args.rank = 0
        nodes = ["127.0.0.1"]

    if args.distributed:
        mp.set_start_method('forkserver')

        print(args.rank)
        port = np.random.randint(15000, 15025)
        args.dist_url = 'tcp://{}:{}'.format(nodes[0], port)
        print(args.dist_url)
        args.dist_backend = 'nccl'
        args.gpu = None

    ngpus_per_node = torch.cuda.device_count()
    args.num_workers = args.workers
    args.ngpus_per_node = ngpus_per_node
    if args.distributed:
        args.world_size = ngpus_per_node * args.world_size
        
    else:
        if ngpus_per_node == 1:
            args.gpu = 0

    optimizer_state_dict=None
    model = {}
    model['UnetAdaptiveBins'] = models.UnetAdaptiveBins.build(n_bins=args.n_bins, min_val=args.min_depth, max_val=args.max_depth,
                                          norm=args.norm)
    if args.pose_model_type == "separate_resnet":
        model["pose_encoder"] = networks.ResnetEncoder(
            args.num_layers,
            args.weights_init == "pretrained",
            num_input_images=args.num_pose_frames
        )
        model["pose"] = networks.PoseDecoder(
            model["pose_encoder"].num_ch_enc,num_input_features=1,num_frames_to_predict_for=2
        )
    elif args.pose_model_type == "posecnn":
        models["pose"] = networks.PoseCNN(args.num_input_frames if args.pose_model_input == "all" else 2)

    if args.gpu is not None:  # If a gpu is set by user: NO PARALLELISM!!
        torch.cuda.set_device(args.gpu)
        for key,item in model.items():
            model[key] = model[key].cuda(args.gpu)
    device = args.gpu
    experiment_name=args.name
    args.epoch = 0
    args.last_epoch = -1
    epochs = args.epochs
    lr=args.lr

    backproject_depth = {}
    project_3d = {}
    
    h = args.input_height
    w = args.input_width

    backproject_depth = BackprojectDepth(args.batch_size, h, w)
    backproject_depth.cuda(device)

    project_3d = Project3D(args.batch_size, h, w)
    project_3d.cuda(device)

    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    params = []
    ###################################### Logging setup #########################################
    print(f"Training {experiment_name}")

    run_id = f"{dt.now().strftime('%d-%h_%H-%M')}-nodebs{args.bs}-tep{epochs}-lr{lr}-wd{args.wd}-{uuid.uuid4()}"
    name = f"{experiment_name}_{run_id}"
    should_write = ((not args.distributed) or args.rank == 0)
    should_log = should_write and logging
    if should_log:
        tags = args.tags.split(',') if args.tags != '' else None
        if args.dataset != 'nyu':
            PROJECT = PROJECT + f"-{args.dataset}"
        wandb.init(project=PROJECT, name=name, config=args, dir=args.root, tags=tags, notes=args.notes)
        # wandb.watch(model)
    ################################################################################################

    # train_loader = DepthDataLoader(args, 'train').data
    # test_loader = DepthDataLoader(args, 'online_eval').data
    train_loader = Seq2DataLoader(args, 'train').data
    test_loader = Seq2DataLoader(args, 'online_eval').data
    ###################################### losses ##############################################
    criterion_ueff = SILogLoss()
    criterion_bins = BinsChamferLoss() if args.chamfer else None
    ################################################################################################
    for key,item in model.items():
        model[key].train()

    ###################################### Optimizer ################################################
    if args.same_lr:
        print("Using same LR")
        for key,item in model.items():
            params += list(model[key].parameters())
    else:
        print("Using diff LR")
        m = model['UnetAdaptiveBins'].module if args.multigpu else model['UnetAdaptiveBins']
        params = [{"params": m.get_1x_lr_params(), "lr": lr / 10},
                  {"params": m.get_10x_lr_params(), "lr": lr}]
        for key,item in model.items():
            if key != 'UnetAdaptiveBins':
                params += [{"params":model[key].parameters(),"lr":args.plr}] # 需要测试

    optimizer = optim.AdamW(params, weight_decay=args.wd, lr=args.lr)
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
    ################################################################################################
    # some globals
    iters = len(train_loader)
    step = args.epoch * iters
    best_loss = np.inf
    if not args.no_ssim:
        args.ssim = SSIM()
        args.ssim.cuda(device)
    ###################################### Scheduler ###############################################
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, lr, epochs=epochs, steps_per_epoch=len(train_loader),
                                              cycle_momentum=True,
                                              base_momentum=0.85, max_momentum=0.95, last_epoch=args.last_epoch,
                                              div_factor=args.div_factor,
                                              final_div_factor=args.final_div_factor)
    if args.resume != '' and scheduler is not None:
        scheduler.step(args.epoch + 1)
    ################################################################################################
    
    # max_iter = len(train_loader) * epochs
    for epoch in range(args.epoch, epochs):
        if should_log: wandb.log({"Epoch": epoch}, step=step)
        for i, batch in tqdm(enumerate(train_loader), desc=f"Epoch: {epoch + 1}/{epochs}. Loop: Train",
                             total=len(train_loader)) if is_rank_zero(
                args) else enumerate(train_loader):
            optimizer.zero_grad()
            # for key, ipt in batch.items():
            #     batch[key] = ipt.to(device)
            if 'has_valid_depth' in batch:
                if not batch['has_valid_depth']:
                    continue
            outputs = {}
            img = batch[('color_tensor',0)].to(device)

            depth = batch['depth_gt'].to(device)
            bin_edges, pred = model["UnetAdaptiveBins"](img)
            outputs["predict_depth"] = pred
            outputs.update(predict_pose(batch,model))
            for i,frame_id in enumerate(args.frame_ids[1:]):
                T = outputs["cam_T_cam",0,frame_id]
                # from the authors of https://arxiv.org/abs/1712.00175
                if args.pose_model_type == "posecnn":
                    axisangle = outputs[("axisangle", 0, frame_id)]
                    translation = outputs[("translation", 0, frame_id)]
                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)
                    T = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)
                cam_points = backproject_depth(depth, batch[("inv_K",0)])
                pix_coords = project_3d(cam_points, batch[("K",0)], T)
                outputs[("sample", frame_id)] = pix_coords
                outputs[("color", frame_id)] = F.grid_sample(batch[("color", frame_id)], outputs[("sample", frame_id)],padding_mode="border")
                if not args.disable_automasking:
                    outputs[("color_identity", frame_id)] = \
                        batch[("color", frame_id)]

            reproject_loss = compute_reprojection_loss(args,batch,outputs) # losses 是一个字典

            mask = depth > args.min_depth
            l_dense = criterion_ueff(pred, depth, mask=mask.to(torch.bool), interpolate=True)

            if args.w_chamfer > 0:
                l_chamfer = criterion_bins(bin_edges, depth)
            else:
                l_chamfer = torch.Tensor([0]).to(img.device)
            loss = l_dense + args.w_chamfer * l_chamfer + reproject_loss
            loss.backward()

            nn.utils.clip_grad_norm_(params, 0.1)  # optional
            optimizer.step()
            if should_log and step % 5 == 0:
                wandb.log({f"Train/{criterion_ueff.name}": l_dense.item()}, step=step)
                wandb.log({f"Train/{criterion_bins.name}": l_chamfer.item()}, step=step)

            step += 1
            scheduler.step()
            
    ########################################################################################################

        if should_write and step % args.validate_every == 0:

            ################################# Validation loop ##################################################
            for m in model.values():
                m.eval()
            metrics, val_si = validate(args, model, test_loader, criterion_ueff, epoch, epochs, device)

            # print("Validated: {}".format(metrics))
            if should_log:
                wandb.log({
                    f"Test/{criterion_ueff.name}": val_si.get_value(),
                    # f"Test/{criterion_bins.name}": val_bins.get_value()
                }, step=step)

                wandb.log({f"Metrics/{k}": v for k, v in metrics.items()}, step=step)
                model_io.save_checkpoint(model, optimizer, epoch, f"{experiment_name}_{run_id}_latest.pt",
                                            root=os.path.join('.', "checkpoints"))

            if metrics['abs_rel'] < best_loss and should_write:
                model_io.save_checkpoint(model, optimizer, epoch, f"{experiment_name}_{run_id}_best.pt",
                                            root=os.path.join('.', "checkpoints"))
                best_loss = metrics['abs_rel']
            for m in model.values():
                m.train()
            #################################################################################################
