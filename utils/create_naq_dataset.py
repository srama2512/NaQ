import argparse
import json
import math
import os
import random

import pandas as pd
import tqdm


def get_em_clips(annotations_dir, split="train", em_version="v1"):
    video_clip_infos = {}
    for task in ["vq", "nlq", "moments"]:
        if em_version == "v1":
            path = os.path.join(annotations_dir, f"{task}_{split}.json")
        else:
            path = os.path.join(annotations_dir, f"{task}_{split}_v2.json")
        with open(path, "r") as f:
            task_data = json.load(f)
        for v in task_data["videos"]:
            video_uid = v["video_uid"]
            for c in v["clips"]:
                clip_uid = c["clip_uid"]
                video_start_sec = c["video_start_sec"]
                video_end_sec = c["video_end_sec"]
                c_info = (video_start_sec, video_end_sec)
                # If clip was already covered in a previous task, skip it
                if (video_uid, clip_uid) in video_clip_infos:
                    assert video_clip_infos[(video_uid, clip_uid)] == c_info
                else:
                    video_clip_infos[(video_uid, clip_uid)] = c_info

    video_uids = set(list(map(lambda k: k[0], video_clip_infos.keys())))
    clip_uids = set(list(map(lambda k: k[1], video_clip_infos.keys())))
    print(f"# Unique videos: {len(video_uids)}")
    print(f"# Unique clips: {len(clip_uids)}")

    video2clips_map = {}
    for (video_uid, clip_uid), (
        video_start_sec,
        video_end_sec,
    ) in video_clip_infos.items():
        if video_uid not in video2clips_map:
            video2clips_map[video_uid] = []
        video2clips_map[video_uid].append(
            {
                "clip_uid": clip_uid,
                "video_start_sec": video_start_sec,
                "video_end_sec": video_end_sec,
            }
        )

    return sorted(list(clip_uids)), video2clips_map


def load_json(path):
    with open(path, "r") as fp:
        data = json.load(fp)
    return data


def save_json(path, data):
    with open(path, "w") as fp:
        json.dump(data, fp)


def load_narrations_data(path):
    # data = pd.read_csv(path, sep='\t', on_bad_lines='skip')
    data = pd.read_csv(path, sep="\t", error_bad_lines=False)
    video_uid_to_narrations = {}

    video_uids = data["video_uid"].tolist()
    clip_starts = data["clip_start"].tolist()
    clip_ends = data["clip_end"].tolist()
    clip_texts = data["clip_text"].tolist()

    for video_uid, start_sec, end_sec, clip_text in tqdm.tqdm(
        zip(video_uids, clip_starts, clip_ends, clip_texts),
        total=len(video_uids),
        desc="Loading narrations data",
    ):
        if video_uid not in video_uid_to_narrations:
            video_uid_to_narrations[video_uid] = []
        video_uid_to_narrations[video_uid].append((start_sec, end_sec, clip_text))
    return video_uid_to_narrations


def create_narrations_data(
    clips,
    video2clips_map,
    video_narrations_data,
    video_fps=30.0,
    split="train",
):
    """Create NLQ-like data from narrations following schema here:
    https://github.com/EGO4D/docs/blob/main/docs/data/annotations-schemas.md
    """
    clips = set(clips)
    em_data = {"videos": []}

    pbar = tqdm.tqdm(desc="Converting narrations data to NLQ format")
    for video_uid, video_narrations in video_narrations_data.items():
        if video_uid not in video2clips_map:
            continue
        clip_info = video2clips_map[video_uid]
        ####################################################################
        # Get annotations for video
        video_data = {
            "video_uid": video_uid,
            "clips": [],
            "split": split,
        }
        for info in clip_info:
            clip_uid = info["clip_uid"]
            if clip_uid not in clips:
                continue
            video_start_sec = info["video_start_sec"]
            video_end_sec = info["video_end_sec"]
            clip_data = {
                "clip_uid": clip_uid,
                "video_start_sec": video_start_sec,
                "video_end_sec": video_end_sec,
                "video_start_frame": int(math.floor(video_start_sec * video_fps)),
                "video_end_frame": int(math.ceil(video_end_sec * video_fps)),
            }
            ####################################################################
            # Get annotations for clip
            language_queries = []
            for narr_start_sec, narr_end_sec, narr_text in video_narrations:
                if (narr_start_sec < video_start_sec) or (narr_end_sec > video_end_sec):
                    continue
                query = {
                    "clip_start_sec": narr_start_sec - video_start_sec,
                    "clip_end_sec": narr_end_sec - video_start_sec,
                    "video_start_sec": narr_start_sec,
                    "video_end_sec": narr_end_sec,
                    "video_start_frame": int(math.floor(narr_start_sec * video_fps)),
                    "video_end_frame": int(math.ceil(narr_end_sec * video_fps)),
                    "query": narr_text,
                }
                language_queries.append(query)
                pbar.update()
            clip_data["annotations"] = [
                {
                    "language_queries": language_queries,
                    "annotation_uid": f"{video_uid}_{clip_uid}",
                }
            ]
            ####################################################################
            video_data["clips"].append(clip_data)
        ####################################################################
        em_data["videos"].append(video_data)

    return em_data


def calculate_stats(data):
    n_queries = 0
    clip_uids = set()
    video_uids = set()
    for v in data["videos"]:
        if len(v["clips"]) > 0:
            video_uids.add(v["video_uid"])
        for c in v["clips"]:
            clip_uids.add(c["clip_uid"])
            for a in c["annotations"]:
                if "language_queries" not in a:
                    continue
                for l in a["language_queries"]:
                    if l is None:
                        continue
                    n_queries += 1
    return n_queries, video_uids, clip_uids


def merge_naq_and_nlq(naq_src, nlq_src):
    assert nlq_src["videos"][0]["split"] == naq_src["videos"][0]["split"]
    split = naq_src["videos"][0]["split"]

    # Get mapping from (video_uid, clip_uid) -> annotations
    nlq_vc2a = {}
    clip_md = {}
    for v in nlq_src["videos"]:
        vid = v["video_uid"]
        for c in v["clips"]:
            cid = c["clip_uid"]
            assert (vid, cid) not in nlq_vc2a
            nlq_vc2a[(vid, cid)] = c["annotations"]
            assert cid not in clip_md
            clip_md[cid] = {
                "video_start_sec": c["video_start_sec"],
                "video_end_sec": c["video_end_sec"],
                "video_start_frame": c["video_start_frame"],
                "video_end_frame": c["video_end_frame"],
            }

    naq_vc2a = {}
    for v in naq_src["videos"]:
        vid = v["video_uid"]
        for c in v["clips"]:
            cid = c["clip_uid"]
            assert (vid, cid) not in naq_vc2a
            naq_vc2a[(vid, cid)] = c["annotations"]
            c_md = {
                "video_start_sec": c["video_start_sec"],
                "video_end_sec": c["video_end_sec"],
                "video_start_frame": c["video_start_frame"],
                "video_end_frame": c["video_end_frame"],
            }
            # Sanity check
            if cid in clip_md:
                for key in ["video_start_sec", "video_end_sec"]:
                    assert math.isclose(clip_md[cid][key], c_md[key])
            else:
                clip_md[cid] = c_md

    # Create new dataset
    tgt_data = {"videos": []}
    vc_tuples = set(list(nlq_vc2a.keys()) + list(naq_vc2a.keys()))
    v2cs = {}
    for vid, cid in vc_tuples:
        if vid not in v2cs:
            v2cs[vid] = []
        v2cs[vid].append(cid)

    for vid in v2cs.keys():
        v_data = {"video_uid": vid, "clips": [], "split": split}
        for cid in v2cs[vid]:
            c_data = {
                "clip_uid": cid,
                **clip_md[cid],
                "annotations": [],
            }
            if (vid, cid) in nlq_vc2a:
                c_data["annotations"] += nlq_vc2a[(vid, cid)]
            if (vid, cid) in naq_vc2a:
                c_data["annotations"] += naq_vc2a[(vid, cid)]
            v_data["clips"].append(c_data)
        tgt_data["videos"].append(v_data)

    # Sanity check
    nlq_n_queries, nlq_video_uids, nlq_clip_uids = calculate_stats(nlq_src)
    naq_n_queries, naq_video_uids, naq_clip_uids = calculate_stats(naq_src)
    tgt_n_queries, tgt_video_uids, tgt_clip_uids = calculate_stats(tgt_data)

    assert tgt_n_queries == nlq_n_queries + naq_n_queries
    assert (nlq_video_uids | naq_video_uids) == tgt_video_uids
    assert (nlq_clip_uids | naq_clip_uids) == tgt_clip_uids

    # Print stats
    print("========> NLQ stats")
    print(f"# queries {nlq_n_queries}")
    print(f"# videos {len(nlq_video_uids)}")
    print(f"# clips {len(nlq_clip_uids)}")

    print("========> NaQ stats")
    print(f"# queries {naq_n_queries}")
    print(f"# videos {len(naq_video_uids)}")
    print(f"# clips {len(naq_clip_uids)}")

    print("========> NLQ+NaQ stats")
    print(f"# queries {tgt_n_queries}")
    print(f"# videos {len(tgt_video_uids)}")
    print(f"# clips {len(tgt_clip_uids)}")

    return tgt_data


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, choices=["nlq", "tacos"], default="nlq")
    parser.add_argument("--em_version", type=str, choices=["v1", "v2"], default="v1")
    args = parser.parse_args()

    random.seed(300)

    naq_root = os.environ["NAQ_ROOT"]
    annotations_dir = os.path.join(naq_root, "data")
    save_dir = os.path.join(naq_root, "data")
    video_narrations_data = load_narrations_data(
        os.path.join(naq_root, "data/egoclip.csv")
    )
    em_clips, em_video2clips_map = get_em_clips(
        annotations_dir, split="train", em_version=args.em_version
    )

    if args.em_version == "v1":
        # Split videos into 95% train and 5% val
        # Note: This splitting is not needed, but is done for reproducibility.
        video_uids = sorted(list(em_video2clips_map.keys()))
        random.shuffle(video_uids)
        n_train = int(len(video_uids) * 0.95)
        train_video_uids = video_uids[:n_train]
    else:
        train_video_uids = sorted(list(em_video2clips_map.keys()))

    em_train_video2clips_map = {v: em_video2clips_map[v] for v in train_video_uids}

    # Create EM data
    naq_train_data = create_narrations_data(
        em_clips, em_train_video2clips_map, video_narrations_data, split="train"
    )

    # Load NLQ data
    nlq_train_data = load_json(os.path.join(annotations_dir, f"{args.type}_train.json"))

    # Merge NLQ+NaQ data
    naq_nlq_train_data = merge_naq_and_nlq(naq_train_data, nlq_train_data)

    if args.em_version == "v1":
        save_path = os.path.join(save_dir, f"{args.type}_aug_naq_train.json")
    else:
        save_path = os.path.join(save_dir, f"{args.type}_aug_naq_train_v2.json")
    save_json(save_path, naq_nlq_train_data)
