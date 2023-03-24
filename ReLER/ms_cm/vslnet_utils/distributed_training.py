import datetime
import os

import torch
import torch.distributed as distrib


DEFAULT_PORT = 8648
DEFAULT_PORT_RANGE = 127
DEFAULT_MAIN_ADDR = "127.0.0.1"


def get_distrib_size():
    # Check to see if we should parse from torch.distributed.launch
    if os.environ.get("LOCAL_RANK", None) is not None:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    # Else parse from SLURM is using SLURM
    elif os.environ.get("SLURM_JOBID", None) is not None:
        local_rank = int(os.environ["SLURM_LOCALID"])
        world_rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
    # Otherwise setup for just 1 process, this is nice for testing
    else:
        local_rank = 0
        world_rank = 0
        world_size = 1

    return local_rank, world_rank, world_size


def ddp_setup():
    assert distrib.is_available(), "torch.distributed must be available"
    local_rank, world_rank, world_size = get_distrib_size()
    slurm_jobid = os.environ.get("SLURM_JOB_ID", None)
    main_port = int(os.environ.get("MAIN_PORT", DEFAULT_PORT))
    if slurm_jobid is not None:
        main_port += int(slurm_jobid) % int(
            os.environ.get("MAIN_PORT_RANGE", DEFAULT_PORT_RANGE)
        )
    main_addr = os.environ.get("MAIN_ADDR", DEFAULT_MAIN_ADDR)
    tcp_store = distrib.TCPStore(
        main_addr,
        main_port,
        world_size,
        world_rank == 0,
        timeout=datetime.timedelta(seconds=18000),
    )
    distrib.init_process_group(
        "nccl",
        store=tcp_store,
        rank=world_rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=18000),
    )
    return local_rank, world_rank, world_size


def ddp_cleanup():
    distrib.destroy_process_group()
