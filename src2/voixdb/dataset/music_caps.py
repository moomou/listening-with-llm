# TODO: generate musicsynth dataset ONLY and train on that
import csv
import multiprocessing
import pathlib as pl
import subprocess

import torch

from .. import util

ROOT = pl.Path("/media/2nvme/data/MusicCaps/")

CSV = ROOT / "musiccaps-public.csv"
CSV_FILTERED = ROOT / "filtered.csv"
AUDIO_DIR = ROOT / "videos"


class MusicCapsDataset(torch.utils.data.Dataset):
    "Characterizes a dataset for PyTorch"

    def __init__(self, entries, tokenizer, prompt_template):
        self.entries = entries
        self.tokenizer = tokenizer
        self.prompt_template = prompt_template

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        "Generates one sample of data"
        entry = self.entries[index]

        mels = util.load_audio_mels(entry["file"])
        caption = entry["caption"]

        prompt_ids, prompt_attention_mask = util.text_2_ids_and_attention_mask(
            self.tokenizer,
            self.prompt_template(),
        )
        end_prompt_ids, end_prompt_attention_mask = util.text_2_ids_and_attention_mask(
            self.tokenizer,
            util.end_template(),
            truncate=True,
        )
        cap_ids, cap_attention_mask = util.text_2_ids_and_attention_mask(
            self.tokenizer,
            caption,
            truncate=True,
        )

        return {
            "_id": [str(entry["file"])],
            "audio_mels": mels.squeeze(0).half(),
            "cap_ids": cap_ids.squeeze(0),
            "cap_attention_mask": cap_attention_mask.squeeze(0),
            "prompt_ids": prompt_ids.squeeze(0),
            "prompt_attention_mask": prompt_attention_mask.squeeze(0),
            "end_prompt_ids": end_prompt_ids.squeeze(0),
            "end_prompt_attention_mask": end_prompt_attention_mask.squeeze(0),
        }


def load_csv(load_raw=False):
    if load_raw:
        entries = []
        mp3s = []
        pool = multiprocessing.Pool(6)

        with open(CSV, mode="r") as csv_file:
            # Create a CSV reader
            csv_reader = csv.DictReader(csv_file)
            # Iterate over each row in the CSV file
            for i, row in enumerate(csv_reader):
                # Each row is a dictionary where the keys are the column names
                file_path = AUDIO_DIR / f"{row['ytid']}-{i}.mp3"
                row["file"] = file_path
                entries.append(row)
                mp3s.append(file_path)

        res = pool.map(verify_mp3, mp3s)
        filtered = []
        for i, is_valid in enumerate(res):
            if is_valid:
                filtered.append(entries[i])

        # save filtered
        # Writing to CSV file
        with open(CSV_FILTERED, mode="w", newline="") as file:
            field_names = list(filtered[0].keys())
            writer = csv.DictWriter(file, fieldnames=field_names)
            # Write header
            writer.writeheader()
            # Write data
            writer.writerows(filtered)

        return filtered
    else:
        with open(CSV_FILTERED, mode="r") as csv_file:
            # Create a CSV reader
            csv_reader = csv.DictReader(csv_file)
            return list(csv_reader)


def verify_mp3(output_path_wt_suffix) -> bool:
    command = ["ffmpeg", "-v", "error", "-i", output_path_wt_suffix, "-f", "null", "-"]
    result = subprocess.run(command, capture_output=True, text=True)
    # Check the return code to see if there were any issues
    return result.returncode == 0
