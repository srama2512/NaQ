import json
import random

import numpy as np
import torch
import torch.utils.data

from utils.data_util import pad_seq, pad_char_seq, pad_video_seq


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset,
        video_features,
        nlq_subsample_clips="",
        narr_subsample_clips="",
        nar_rand_exp_factor=-1.0,
        nar_rand_translate=False,
    ):
        super(Dataset, self).__init__()
        self.dataset = self.get_subsampled_dataset(
            dataset,
            narr_subsample_clips,
            nlq_subsample_clips,
        )
        self.video_features = video_features
        self.nar_rand_exp_factor = nar_rand_exp_factor
        self.nar_rand_translate = nar_rand_translate
        if self.nar_rand_exp_factor > 0:
            assert self.nar_rand_exp_factor > 1

    def __getitem__(self, index):
        record = self.dataset[index]
        video_feature = self.video_features[record["vid"]]
        s_ind, e_ind = int(record["s_ind"]), int(record["e_ind"])
        s_ind, e_ind = self.expand_narration_window(
            record["query"], s_ind, e_ind, video_feature.shape[0]
        )
        s_ind, e_ind = self.validate(s_ind, e_ind, video_feature.shape[0])
        word_ids = record["w_ids"]
        char_ids = record.get("c_ids", None)
        return record, video_feature, word_ids, char_ids, s_ind, e_ind

    def __len__(self):
        return len(self.dataset)

    def get_subsampled_dataset(self, dataset, narr_ss_clips, nlq_ss_clips):
        if narr_ss_clips == "" and nlq_ss_clips == "":
            return dataset
        narr_dataset, nlq_dataset = self.split_data_into_narrations_and_nlq(dataset)
        narr_dataset_ss = self.get_subsampled_clips(
            narr_dataset, narr_ss_clips, "Narrations"
        )
        nlq_dataset_ss = self.get_subsampled_clips(nlq_dataset, nlq_ss_clips, "NLQ")
        dataset_ss = nlq_dataset_ss + narr_dataset_ss
        # Sanity checks
        n_nlq, n_narr = self.get_nlq_narr_stats(dataset)
        n_nlq_ss, n_narr_ss = self.get_nlq_narr_stats(dataset_ss)
        print(
            "====> # NLQ queries: {}/{} | # Narr queries: {}/{}".format(
                n_nlq_ss, n_nlq, n_narr_ss, n_narr
            )
        )
        return dataset_ss

    def get_nlq_narr_stats(self, dataset):
        n_nlq_queries = 0
        n_narr_queries = 0
        for record in dataset:
            if record["query"].startswith("#"):
                n_narr_queries += 1
            else:
                n_nlq_queries += 1
        return n_nlq_queries, n_narr_queries

    def split_data_into_narrations_and_nlq(self, dataset):
        narr_dataset = [record for record in dataset if record["query"].startswith("#")]
        nlq_dataset = [
            record for record in dataset if not record["query"].startswith("#")
        ]
        assert len(narr_dataset) + len(nlq_dataset) == len(dataset)
        return narr_dataset, nlq_dataset

    def get_subsampled_clips(self, dataset, ss_clips_path, mode):
        if ss_clips_path == "":
            return dataset
        unique_clips = set([d["vid"] for d in dataset])
        ss_clips = json.load(open(ss_clips_path, "r"))
        ss_dataset = [record for record in dataset if record["vid"] in ss_clips]
        print_str = f"======> Subsampling {mode} data"
        print_str += "\n" + "Sampling {} / {} clips".format(
            len(ss_clips), len(unique_clips)
        )
        print_str += "\n" + "Sampling {} / {} queries".format(
            len(ss_dataset), len(dataset)
        )
        print(print_str)
        return ss_dataset

    def validate(self, s, e, maxlen):
        s = int(np.clip(s, 0, maxlen - 1).item())
        e = int(np.clip(e, 0, maxlen - 1).item())
        if s > e:
            print("Invalid: (s, e) = {}, {}".format(s, e))
        assert s <= e
        return s, e

    def expand_narration_window(self, query, s, e, maxlen):
        if query.startswith("#") and self.nar_rand_exp_factor > 0:
            c = (s + e) / 2.0
            w = e - s + 1
            w_scale = random.uniform(1.0, self.nar_rand_exp_factor)
            w_ = w * w_scale
            t = 0
            if self.nar_rand_translate:
                t = random.uniform(-(w_ - w) / 2.0, (w_ - w) / 2.0)
            c = c + t
            s = max(int(np.rint(c - w_ / 2.0)), 0)
            e = min(int(np.rint(c + w_ / 2.0)), maxlen - 1)

        return s, e


def train_collate_fn(data):
    records, video_features, word_ids, char_ids, s_inds, e_inds = zip(*data)
    # If BERT is used, pad individual components of the dictionary.
    if not isinstance(word_ids[0], list):
        pad_input_ids, _ = pad_seq([ii["input_ids"] for ii in word_ids])
        pad_attention_mask, _ = pad_seq([ii["attention_mask"] for ii in word_ids])
        if "token_type_ids" in word_ids[0]:
            pad_token_type_ids, _ = pad_seq([ii["token_type_ids"] for ii in word_ids])
        word_ids_new = {
            "input_ids": torch.LongTensor(pad_input_ids),
            "attention_mask": torch.LongTensor(pad_attention_mask),
        }
        if "token_type_ids" in word_ids[0]:
            word_ids_new["token_type_ids"] = torch.LongTensor(pad_token_type_ids)
        word_ids = word_ids_new
        char_ids = None
    else:
        # process word ids
        word_ids, _ = pad_seq(word_ids)
        word_ids = np.asarray(word_ids, dtype=np.int32)  # (batch_size, w_seq_len)
        # process char ids
        char_ids, _ = pad_char_seq(char_ids)
        char_ids = np.asarray(
            char_ids, dtype=np.int32
        )  # (batch_size, w_seq_len, c_seq_len)
        word_ids = torch.tensor(word_ids, dtype=torch.int64)
        char_ids = torch.tensor(char_ids, dtype=torch.int64)
    # process video features
    vfeats, vfeat_lens = pad_video_seq(video_features)
    vfeats = np.asarray(vfeats, dtype=np.float32)  # (batch_size, v_seq_len, v_dim)
    vfeat_lens = np.asarray(vfeat_lens, dtype=np.int32)  # (batch_size, )
    # process labels
    max_len = np.max(vfeat_lens)
    batch_size = vfeat_lens.shape[0]
    s_labels = np.asarray(s_inds, dtype=np.int64)
    e_labels = np.asarray(e_inds, dtype=np.int64)
    h_labels = np.zeros(shape=[batch_size, max_len], dtype=np.int32)
    extend = 0.1
    for idx in range(batch_size):
        st, et = s_inds[idx], e_inds[idx]
        cur_max_len = vfeat_lens[idx]
        extend_len = round(extend * float(et - st + 1))
        if extend_len > 0:
            st_ = max(0, st - extend_len)
            et_ = min(et + extend_len, cur_max_len - 1)
            h_labels[idx][st_ : (et_ + 1)] = 1
        else:
            h_labels[idx][st : (et + 1)] = 1
    # convert to torch tensor
    vfeats = torch.tensor(vfeats, dtype=torch.float32)
    vfeat_lens = torch.tensor(vfeat_lens, dtype=torch.int64)
    s_labels = torch.tensor(s_labels, dtype=torch.int64)
    e_labels = torch.tensor(e_labels, dtype=torch.int64)
    h_labels = torch.tensor(h_labels, dtype=torch.int64)
    return records, vfeats, vfeat_lens, word_ids, char_ids, s_labels, e_labels, h_labels


def test_collate_fn(data):
    records, video_features, word_ids, char_ids, *_ = zip(*data)
    # If BERT is used, pad individual components of the dictionary.
    if not isinstance(word_ids[0], list):
        pad_input_ids, _ = pad_seq([ii["input_ids"] for ii in word_ids])
        pad_attention_mask, _ = pad_seq([ii["attention_mask"] for ii in word_ids])
        if "token_type_ids" in word_ids[0]:
            pad_token_type_ids, _ = pad_seq([ii["token_type_ids"] for ii in word_ids])
        word_ids_new = {
            "input_ids": torch.LongTensor(pad_input_ids),
            "attention_mask": torch.LongTensor(pad_attention_mask),
        }
        if "token_type_ids" in word_ids[0]:
            word_ids_new["token_type_ids"] = torch.LongTensor(pad_token_type_ids)
        word_ids = word_ids_new
        char_ids = None
    else:
        # process word ids
        word_ids, _ = pad_seq(word_ids)
        word_ids = np.asarray(word_ids, dtype=np.int32)  # (batch_size, w_seq_len)
        # process char ids
        char_ids, _ = pad_char_seq(char_ids)
        char_ids = np.asarray(
            char_ids, dtype=np.int32
        )  # (batch_size, w_seq_len, c_seq_len)
        word_ids = torch.tensor(word_ids, dtype=torch.int64)
        char_ids = torch.tensor(char_ids, dtype=torch.int64)
    # process video features
    vfeats, vfeat_lens = pad_video_seq(video_features)
    vfeats = np.asarray(vfeats, dtype=np.float32)  # (batch_size, v_seq_len, v_dim)
    vfeat_lens = np.asarray(vfeat_lens, dtype=np.int32)  # (batch_size, )
    # convert to torch tensor
    vfeats = torch.tensor(vfeats, dtype=torch.float32)
    vfeat_lens = torch.tensor(vfeat_lens, dtype=torch.int64)
    return records, vfeats, vfeat_lens, word_ids, char_ids


def get_train_loader(dataset, video_features, configs, is_distributed=False):
    train_set = Dataset(
        dataset=dataset,
        video_features=video_features,
        narr_subsample_clips=configs.scaling_subsample_clips,
        nlq_subsample_clips=configs.zeroshot_subsample_clips,
        nar_rand_exp_factor=configs.nar_rand_window_expansion_factor,
        nar_rand_translate=configs.nar_rand_window_translate,
    )
    train_sampler = None
    if is_distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=configs.batch_size,
        shuffle=(train_sampler is None),
        pin_memory=True,
        num_workers=configs.data_loader_workers,
        collate_fn=train_collate_fn,
        sampler=train_sampler,
    )
    print(f"====> Len of data loader: {len(train_loader) * configs.batch_size}")
    print(f"====> Len of data set: {len(train_set)}")
    return train_loader, train_sampler


def get_test_loader(dataset, video_features, configs, is_distributed=False):
    test_set = Dataset(dataset=dataset, video_features=video_features)
    test_sampler = None
    if is_distributed:
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            test_set, shuffle=False
        )
    batch_size = (
        configs.batch_size if configs.eval_batch_size == -1 else configs.eval_batch_size
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=configs.data_loader_workers,
        collate_fn=test_collate_fn,
        sampler=test_sampler,
    )
    return test_loader, test_sampler
