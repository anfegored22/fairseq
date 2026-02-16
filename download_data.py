import random

import torch
import webdataset as wds
from torchcodec.decoders import AudioDecoder

from .urls import build_urls

"""
Shared utilities for ups_challenge datasets and training.
"""
import torch

def build_lid_index(index_path: str = "./data/lid_index.pkl", hf_token: str = None):
    """
    Build a language ID index from the JSONL results file.
    The index maps (tar_number, filename) to predicted language.
    Saves the index as a pickle file.

    Args:
        index_path (str): Path to save the index pickle file.
    """

    if hf_token is None:
        hf_token = os.getenv("HF_TOKEN")
    if hf_token is None:
        raise ValueError("HF_TOKEN is not set")

    lid_folder = Path(index_path).parent

    if not os.path.exists(lid_folder):
        os.makedirs(lid_folder)

    if not os.path.exists(lid_folder / "lang_id_results.jsonl"):
        # Download the results file from Hugging Face
        import requests

        url = "https://huggingface.co/datasets/MLCommons/unsupervised_peoples_speech/resolve/main/lang_id_results.jsonl"
        headers = {"Authorization": f"Bearer {hf_token}"}
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            raise ValueError(
                f"Failed to download lid_results.jsonl: {response.status_code}"
            )
        with open(lid_folder / "lang_id_results.jsonl", "wb") as f:
            f.write(response.content)
        print(f"Downloaded  lang_id_results.jsonl to {lid_folder / 'lang_id_results.jsonl'}")

    index = {}

    with open(lid_folder / "lang_id_results.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            tar_number = obj["tar_number"]
            filename = os.path.basename(obj["filepath"])
            lang = obj["prediction"]

            index[(tar_number, filename)] = lang

    print(f"Built index with {len(index)} entries")

    with open(index_path, "wb") as f:
        pickle.dump(index, f, protocol=4)

    print(f"Saved to {index_path}")

    build_lid_index_splits(index_path)

def build_urls(
    langs: list[str] = [],
    index_path: str = "./data/lid_index.pkl",
    hf_token: str = None,
) -> list[str]:
    """
    Build a list of WebDataset URLs for the given languages.
    If langs is empty, all languages are included.
    Args:
        langs (list): List of language codes to include. If empty, all languages are included.
        index_path (str): Path to the language ID index folder.
    Returns:
        list[str]: List of WebDataset URLs.
    """

    if hf_token is None:
        hf_token = os.getenv("HF_TOKEN")
    if hf_token is None:
        raise ValueError("HF_TOKEN is not set")
    token = f"Authorization:Bearer {hf_token}"
    if not os.path.exists(index_path):
        build_lid_index(index_path, hf_token=hf_token)

    with open(index_path, "rb") as f:
        lid_index = pickle.load(f)

    all_relevant_tar_numbers = set()
    for (tar_number, _), lang in tqdm(lid_index.items()):
        if len(langs) == 0 or lang in langs:
            all_relevant_tar_numbers.add(tar_number)
    all_relevant_tar_numbers = list(all_relevant_tar_numbers)
    urls = []
    for tar_number in all_relevant_tar_numbers:
        if int(tar_number) <= 5000:
            urls.append(
                f"https://huggingface.co/datasets/MLCommons/unsupervised_peoples_speech/resolve/main/audio/{tar_number}.tar?download=True"
            )
        else:
            urls.append(
                f"https://huggingface.co/datasets/MLCommons/unsupervised_peoples_speech/resolve/main/audio2/{tar_number}.tar?download=True"
            )
    urls = [f"pipe:curl -s -L {url} -H {token}" for url in urls]
    return urls
class LimitedDataset(torch.utils.data.IterableDataset):  # pylint: disable=abstract-method
    """
    Limit the number of valid (non-None) samples from an iterable dataset.
    Use with streaming WebDataset pipelines to cap how many samples are
    consumed (e.g. for quick experiments or balanced splits).
    """

    def __init__(self, dataset, max_samples):
        super().__init__()
        self.dataset = dataset
        self.max_samples = max_samples

    def __iter__(self):
        count = 0
        for sample in self.dataset:
            if sample is not None:
                yield sample
                count += 1
                if count >= self.max_samples:
                    break

def decode_and_normalize(
    sample,
    target_sr=16000,
    chunk_sec=8.192,
    max_chunks_per_example=16,
    shuffle_chunks=False,
):
    """
    sample comes from .to_tuple('mp3', '__key__', '__url__')
    so it's (mp3_bytes, key, url).

    We:
      - decode mp3 using torchaudio
      - resample to default_sample_rate
      - convert to mono
      - return a dict

    Any samples that fail to decode are logged and skipped.
    """
    mp3_bytes, _, _ = sample
    # 8.192s at 16kHz gives 131072 samples (2^17), so T is power-of-two.
    chunk_samples = int(chunk_sec * target_sr)

    output_chunks = []

    decoder = AudioDecoder(source=mp3_bytes, sample_rate=target_sr, num_channels=1)

    duration = decoder.metadata.duration_seconds_from_header

    # Decode once, then slice many chunks from memory; this avoids repeatedly
    # paying MP3 decode cost for each get_samples_played_in_range call.
    samples = decoder.get_samples_played_in_range(0.0, duration)
    waveform = samples.data.squeeze(0)

    def extract_chunk(start_sec):
        start_idx = int(start_sec * target_sr)
        end_idx = start_idx + chunk_samples
        chunk = waveform[start_idx:end_idx]

        if chunk.shape[-1] < chunk_samples:
            pad = chunk_samples - chunk.shape[-1]
            chunk = torch.nn.functional.pad(chunk, (0, pad))

        return chunk

    # ---- 2) If short file, take a single chunk from start ----
    if duration <= chunk_sec:
        chunk = extract_chunk(0.0)

        output_chunks.append(chunk)
        batch_wave = torch.stack(output_chunks)
        attention_mask = torch.ones_like(batch_wave, dtype=torch.long)
        return {
            "input_values": batch_wave,  # [N_chunks, chunk_samples]
            "attention_mask": attention_mask,  # same shape
        }

    # ---- 3) Choose random chunk start times (in seconds) ----
    max_start_sec = duration - chunk_sec

    # Generate random starting times
    start_times = [
        random.uniform(0.0, max_start_sec) for _ in range(max_chunks_per_example)
    ]

    # ---- 4) Slice each chunk from decoded waveform ----
    for start_sec in start_times:
        output_chunks.append(extract_chunk(start_sec))

    # ---- 5) Shuffle chunks across examples ----
    if shuffle_chunks:
        random.shuffle(output_chunks)

    # ---- 6) Stack into batch tensors ----
    batch_wave = torch.stack(output_chunks)

    attention_mask = torch.ones_like(batch_wave, dtype=torch.long)

    return {
        "input_values": batch_wave,  # [N_chunks, chunk_samples]
        "attention_mask": attention_mask,  # same shape
    }


def collate_fn(batch: list):
    """
    Custom collate function to:
    - handle None samples
    - concatenate input_values and attention_masks across the batch dimension
    """
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    input_values = [b["input_values"] for b in batch]

    attention_masks = [b["attention_mask"] for b in batch]
    return {
        "input_values": torch.cat(input_values, dim=0),  # (sum_N_chunks, T)
        "attention_mask": torch.cat(attention_masks, dim=0),
    }


def build_wds_dataset(
    langs: list = [],
    index_path: str = "./data/lid_index.pkl",
    hf_token: str = None,
    max_samples: int = None,
):
    """
    Build a WebDataset dataset for the given languages.
    If langs is empty, all languages are included.

    Args:
        langs: List of language codes to include. If empty, all languages are included.
        index_path: Path to the language ID index folder.
        hf_token: HuggingFace token for dataset access.
        max_samples: Maximum number of valid samples to yield (None for no limit).

    Returns:
        WebDataset (or LimitedDataset wrapping it) yielding decoded audio dicts.
    """
    urls = build_urls(langs, index_path=index_path, hf_token=hf_token)
    dataset = (
        wds.WebDataset(
            urls,
            shardshuffle=False,
        )
        .to_tuple("mp3", "__key__", "__url__", handler=wds.handlers.ignore_and_continue)
        .map(decode_and_normalize)
    )
    if max_samples is not None:
        dataset = LimitedDataset(dataset, max_samples)
    return dataset
