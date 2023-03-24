import copy
import json
import logging
import os
import pprint
import random
import time
from collections import defaultdict

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as distrib
import torch.nn as nn
from easydict import EasyDict as edict
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from utils.basic_utils import AverageMeter, dict_to_markdown
from utils.model_utils import count_parameters

import ms_cm.vslnet_utils.evaluate_ego4d_nlq as ego4d_eval
from ms_cm.configs import BaseOptions
from ms_cm.inference_ego4d_slowfast import eval_epoch, start_inference, setup_model
from ms_cm.sw_vs_ego4d_dataset import Ego4d_dataset, Ego4d_collate, prepare_batch_inputs
from ms_cm.vslnet_utils.data_gen_ego4d import gen_or_load_dataset
from ms_cm.vslnet_utils.data_util import (
    load_json,
    load_lines,
    load_pickle,
    save_pickle,
    time_to_index,
)
from ms_cm.vslnet_utils.distributed_training import get_distrib_size, ddp_setup
from ms_cm.vslnet_utils.runner_utils import filter_checkpoints, eval_test

torch.set_num_threads(1)


def extract_index(start_logits, end_logits):
    start_prob = nn.Softmax(dim=1)(start_logits)
    end_prob = nn.Softmax(dim=1)(end_logits)
    outer = torch.matmul(start_prob.unsqueeze(dim=2), end_prob.unsqueeze(dim=1))
    outer = torch.triu(outer, diagonal=0)

    # Get top 5 start and end indices.
    batch_size, height, width = outer.shape
    outer_flat = outer.view(batch_size, -1)
    _, flat_indices = outer_flat.topk(5, dim=-1)
    start_indices = flat_indices // width
    end_indices = flat_indices % width
    return start_indices, end_indices


def index_to_time(start_index, end_index, num_units, duration):
    s_times = np.arange(0, num_units).astype(np.float32) * duration / float(num_units)
    e_times = (
        np.arange(1, num_units + 1).astype(np.float32) * duration / float(num_units)
    )
    start_time = s_times[start_index]
    end_time = e_times[end_index]
    return start_time, end_time


def set_seed(seed, use_cuda=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)


def train_epoch(
    model,
    criterion,
    train_loader,
    optimizer,
    opt,
    epoch_i,
    global_step,
    tb_writer,
    logger,
    is_distributed,
    local_rank,
    world_rank,
    world_size,
    gt_json_path="../../Datasets/Ego4d/ego4d_annotation/nlq_train.json",
):

    if world_rank == 0:
        logger.info(f"[Epoch {epoch_i+1}]")
    model.train()
    criterion.train()

    # init meters
    time_meters = defaultdict(AverageMeter)
    loss_meters = defaultdict(AverageMeter)

    num_training_examples = len(train_loader)
    timer_dataloading = time.time()
    predictions = []
    if world_rank == 0:
        iterator = tqdm(
            train_loader,
            desc=f"Epoch: {epoch_i}",
            total=num_training_examples,
        )
    else:
        iterator = train_loader
    for batch_idx, batch in enumerate(iterator):
        global_step += 1
        query_record = batch[0]
        time_meters["dataloading_time"].update(time.time() - timer_dataloading)

        timer_start = time.time()
        model_inputs, targets = prepare_batch_inputs(
            batch[1],
            opt.device,
            non_blocking=opt.pin_memory,
            cur_scale=batch[2],
        )
        time_meters["prepare_inputs_time"].update(time.time() - timer_start)

        outputs = model(**model_inputs)

        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(
            loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict
        )
        time_meters["model_forward_time"].update(time.time() - timer_start)

        timer_start = time.time()
        optimizer.zero_grad()
        losses.backward()
        if opt.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
        optimizer.step()
        time_meters["model_backward_time"].update(time.time() - timer_start)

        loss_dict["loss_overall"] = float(losses)  # for logging only
        for k, v in loss_dict.items():
            loss_meters[k].update(
                float(v) * weight_dict[k] if k in weight_dict else float(v)
            )

        timer_dataloading = time.time()

        if global_step % opt.period == 0:

            loss_key_vals = [(k, v.avg) for k, v in loss_meters.items()]
            loss_keys, loss_vals = zip(*loss_key_vals)
            if is_distributed:
                # Synchronize stats across workers
                stats_to_sync = torch.Tensor(loss_vals).to(opt.device)
                distrib.all_reduce(stats_to_sync)
                stats_to_sync = stats_to_sync / world_size
                loss_vals = stats_to_sync.cpu().numpy().tolist()
            if world_rank == 0:
                # print/add logs
                tb_writer.add_scalar(
                    "Train/lr", float(optimizer.param_groups[0]["lr"]), global_step
                )
                tb_writer.add_scalar("Train/epoch", epoch_i, global_step)
                for k, v in zip(loss_keys, loss_vals):
                    tb_writer.add_scalar("Train/{}".format(k), v, global_step)
                to_write = opt.train_log_txt_formatter.format(
                    time_str=time.strftime("%Y_%m_%d_%H_%M_%S"),
                    epoch=epoch_i + 1,
                    loss_str=" ".join(
                        ["{} {:.4f}".format(k, v) for k, v in zip(loss_keys, loss_vals)]
                    ),
                )
                with open(opt.train_log_filepath, "a") as f:
                    f.write(to_write)

    if world_rank == 0:
        logger.info("Epoch time stats:")
        for name, meter in time_meters.items():
            d = {k: f"{getattr(meter, k):.4f}" for k in ["max", "min", "avg"]}
            logger.info(f"{name} ==> {d}")

    return global_step


def train(
    model,
    criterion,
    optimizer,
    lr_scheduler,
    train_dataset,
    val_dataset,
    opt,
    logger,
    is_distributed,
    local_rank,
    world_rank,
    world_size,
):
    best_metric = -1.0
    score_writer = None
    tb_writer = None
    if world_rank == 0:
        score_writer = open(
            os.path.join(opt.results_dir, "eval_results.txt"),
            mode="w",
            encoding="utf-8",
        )
        tb_writer = SummaryWriter(opt.tensorboard_log_dir)
        tb_writer.add_text(
            "hyperparameters", dict_to_markdown(vars(opt), max_str_len=None)
        )
    opt.train_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str}\n"
    opt.eval_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str} [Metrics] {eval_metrics_str}\n"

    train_sampler = None
    if is_distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        collate_fn=Ego4d_collate,
        batch_size=opt.bsz,
        num_workers=opt.num_workers,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        pin_memory=opt.pin_memory,
    )

    # Avoid distributed evaluation
    eval_loader = DataLoader(
        val_dataset,
        collate_fn=Ego4d_collate,
        batch_size=opt.eval_bsz,
        num_workers=opt.num_workers,
        shuffle=False,
        pin_memory=opt.pin_memory,
    )
    es_cnt = 0
    if opt.start_epoch is None:
        start_epoch = -1 if opt.eval_untrained else 0
    else:
        start_epoch = opt.start_epoch
    global_step = 0
    for epoch_i in range(start_epoch, opt.n_epoch):
        if is_distributed:
            train_sampler.set_epoch(epoch_i)

        if epoch_i > -1:
            global_step = train_epoch(
                model,
                criterion,
                train_loader,
                optimizer,
                opt,
                epoch_i,
                global_step,
                tb_writer,
                logger,
                is_distributed,
                local_rank,
                world_rank,
                world_size,
            )
            if opt.optim_name == "AdamW":
                lr_scheduler.step()

        eval_epoch_interval = 1
        # Evaluate only within rank 0
        if world_rank == 0 and (epoch_i + 1) % eval_epoch_interval == 0:
            print(f"\nEpoch: {epoch_i + 1:2d} |", flush=True)
            result_save_path = os.path.join(
                opt.results_dir,
                f"{epoch_i}_preds.json",
            )
            model.eval()
            with torch.no_grad():
                results, mIoU, (score_str, score_dict) = eval_epoch(
                    model,
                    eval_loader,
                    opt,
                    result_save_path,
                    opt.eval_gt_json,
                    epoch_i,
                    tb_writer,
                    return_results_dict=True,
                )
            for name, value in score_dict.items():
                tb_writer.add_scalar(f"Val/{name}", value, global_step)

            print(score_str, flush=True)
            score_writer.write(score_str)
            score_writer.flush()

            if opt.optim_name == "AdamW":
                checkpoint = {
                    "model": model.module.state_dict()
                    if is_distributed
                    else model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch_i,
                    "opt": opt,
                }
            elif opt.optim_name == "BertAdam":
                checkpoint = {
                    "model": model.module.state_dict()
                    if is_distributed
                    else model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch_i,
                    "opt": opt,
                }
            print("_{:0>4d}.ckpt".format(epoch_i))
            # Recall@1, 0.3 IoU overlap --> best metric.
            if results[0][0] >= best_metric:
                best_metric = results[0][0]
                torch.save(
                    checkpoint,
                    os.path.join(
                        opt.results_dir,
                        "model_{}.t7".format(global_step),
                    ),
                )
                # only keep top-3 model checkpoints
                filter_checkpoints(opt.results_dir, suffix="t7", max_to_keep=3)

    if world_rank == 0:
        tb_writer.close()
        score_writer.close()


def start_training(is_distributed, local_rank, world_rank, world_size):
    base_opt = BaseOptions()
    opt = base_opt.parse()
    if world_rank == 0:
        base_opt.display_save(opt)

    if world_rank == 0:
        logger = logging.getLogger(__name__)
        logging.basicConfig(
            format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            level=logging.INFO,
        )
        logger.info("Setup config, data and model...")
    else:
        logger = None

    if is_distributed:
        assert opt.bsz % world_size == 0
        opt.bsz = opt.bsz // world_size
        if world_rank == 0:
            logger.info(f"=====> Setting batch_size to {opt.bsz}")
    opt.device = torch.device(f"cuda:{local_rank}")

    set_seed(opt.seed + world_rank)
    torch.cuda.set_device(local_rank)
    torch.cuda.empty_cache()

    dataset_config = dict(
        v_feat_dirs=opt.v_feat_dirs,
        multiscale_list=opt.numscale_list,
        use_sw=opt.use_sw,
        sw_len_ratio=opt.sw_len_ratio,
        use_vs=opt.use_vs,
        vs_prob=opt.vs_prob,
        max_q_l=opt.max_q_l,
        max_v_l=opt.max_v_l,
        ctx_mode=opt.ctx_mode,
        normalize_v=not opt.no_norm_vfeat,
        normalize_t=not opt.no_norm_tfeat,
        txt_drop_ratio=opt.txt_drop_ratio,
        nar_rand_exp_factor=opt.nar_rand_window_expansion_factor,
        nar_rand_translate=opt.nar_rand_window_translate,
    )
    vslnet_datasetconfigs = edict(
        {
            "task": opt.vsldataset_task,
            "fv": opt.vsldataset_fv,
            "max_pos_len": opt.max_v_l,
            "num_workers": opt.vsldataset_num_workers,
            "vslnet_datapath": opt.vslnet_datapath,
            "save_dir": opt.vslnet_dataset_save_dir,
            "thres_in_train": opt.vslnet_thres_in_train,
        }
    )
    # preparing ego4d dataset by using vslnet configuration
    if is_distributed:
        # prepare dataset only on rank 0
        if world_rank == 0:
            print("********************************************")
            print("******* Preparing dataset on rank 0 ********")
            print("********************************************")
            dataset = gen_or_load_dataset(vslnet_datasetconfigs)
        torch.distributed.barrier()
        # load prepared dataset on remaining ranks
        if world_rank != 0:
            print("********************************************")
            print("****** Preparing dataset on ranks > 0 ******")
            print("********************************************")
            dataset = gen_or_load_dataset(vslnet_datasetconfigs)
    else:
        dataset = gen_or_load_dataset(vslnet_datasetconfigs)

    dataset_config["dataset"] = dataset["train_set"]
    dataset_config["mode"] = "train"
    train_dataset = Ego4d_dataset(**dataset_config)

    dataset_config["dataset"] = dataset["val_set"]
    dataset_config["mode"] = "eval"
    dataset_config["txt_drop_ratio"] = 0
    dataset_config["nar_rand_exp_factor"] = -1.0
    dataset_config["nar_rand_translate"] = False
    eval_dataset = Ego4d_dataset(**dataset_config)

    model, criterion, optimizer, lr_scheduler = setup_model(
        opt, is_distributed, local_rank, world_rank
    )
    if world_rank == 0:
        logger.info(f"Model {model}")
        count_parameters(model)
        logger.info("Start Training...")
    train(
        model,
        criterion,
        optimizer,
        lr_scheduler,
        train_dataset,
        eval_dataset,
        opt,
        logger,
        is_distributed,
        local_rank,
        world_rank,
        world_size,
    )


if __name__ == "__main__":
    is_distributed = get_distrib_size()[2] > 1
    local_rank, world_rank, world_size = 0, 0, 1
    if is_distributed:
        local_rank, world_rank, world_size = ddp_setup()
    # Setup DDP
    start_training(is_distributed, local_rank, world_rank, world_size)
