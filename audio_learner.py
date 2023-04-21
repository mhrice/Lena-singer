import utils.utils as utils
import os
from egs.visinger2.models import SynthesizerTrn
from text import npu
import numpy as np
import torch
from pedalboard import (
    Pedalboard,
    HighpassFilter,
    LowpassFilter,
    Compressor,
    Reverb,
    Distortion,
    Gain,
)
import pyloudnorm as pyln

from collections import deque

MAX_SONGS = 5
current_song = 0
name = "雨打湿了天空灰得更彻底"

with open("test.txt") as data_file:
    data = data_file.readlines()


def parse_label(hps, phos, pitchids, durs, slurs):
    phos = [npu.symbol_converter.ttsing_phone_to_int[pho] for pho in phos]
    pitchs = [
        npu.symbol_converter.ttsing_opencpop_pitch_to_int[pitchid]
        for pitchid in pitchids
    ]

    phos = np.asarray(phos, dtype=np.int32)
    pitchs = np.asarray(pitchs, dtype=np.int32)
    durs = np.asarray(durs, dtype=np.float32)
    slurs = np.asarray(slurs, dtype=np.int32)

    phos = torch.LongTensor(phos)
    pitchs = torch.LongTensor(pitchs)
    durs = torch.FloatTensor(durs)
    slurs = torch.LongTensor(slurs)
    return phos, pitchs, durs, slurs


def load_model(model_dir):
    # load config and model
    model_path = utils.latest_checkpoint_path(model_dir)
    config_path = os.path.join(model_dir, "config.json")

    hps = utils.get_hparams_from_file(config_path)

    print("Load model from : ", model_path)
    print("config: ", config_path)

    net_g = SynthesizerTrn(hps)
    _ = net_g.eval()
    _ = utils.load_checkpoint(model_path, net_g, None)
    return net_g, hps


def pre_generation_processing(phones, notes, dur, mistakes):
    # Overall Speed
    speed = np.random.normal(1, mistakes / 100)
    for i in range(len(dur)):
        dur[i] = dur[i] * speed
    # Phoneme Speed Change
    phone_speedchange_prob = mistakes / 100
    for i in range(len(dur)):
        if np.random.rand() < phone_speedchange_prob:
            dur[i] = dur[i] * np.random.uniform(0.6, 1.4)
    # Random phoneme replacement
    phone_replace_prob = (mistakes / 100) / 2
    for i in range(len(phones)):
        if phones[i] != "SP" and phones[i] != "AP":
            if np.random.random() < phone_replace_prob:
                phones[i] = np.random.choice(
                    list(npu.symbol_converter.ttsing_phone_to_int.keys())
                )
    return phones, notes, dur


def post_generation_processing(audio, mistakes, confidence, sample_rate):
    # Mistakes-based effects
    normalize = LoudnessNormalize(sample_rate, target_lufs_db=-5)

    board = Pedalboard()
    highpass = HighpassFilter(mistakes * 20)  # 0 - 2000
    board.append(highpass)
    lowpass = LowpassFilter(4000 + ((100 - mistakes) * 100))  # 4000 - 14000
    board.append(lowpass)
    apply_distortion = np.random.rand() < (mistakes / 100)

    compressor = Compressor(-mistakes / 10)  # 0 - -10,
    board.append(compressor)
    # Reverb is "good" effect, added with confidence
    dry_wet = (confidence / 100) / 25
    reverb = Reverb(0.7, 0.5, dry_wet, 0.4)
    board.append(reverb)
    # Gain is "good" effect, changed with confidence
    gain = Gain((confidence / 100) * 8 - 4)  # -4 - 4
    board.append(gain)
    if apply_distortion:
        distortion = Distortion(drive_db=np.random.uniform(-40, -60))
        board.append(distortion)

    audio = board(audio.astype(float), sample_rate=sample_rate)
    audio = normalize(audio)
    return audio


def generate_audio(model, pho, pitchid, dur, slur, detune_prob=0):
    with torch.no_grad():
        # data
        pho_lengths = torch.LongTensor([pho.size(0)])
        pho = pho.unsqueeze(0)
        pitchid = pitchid.unsqueeze(0)
        dur = dur.unsqueeze(0)
        slur = slur.unsqueeze(0)

        # infer
        o, _, _ = model.infer(
            pho, pho_lengths, pitchid, dur, slur, random_detune_prob=detune_prob
        )
        audio = o[0, 0].data.cpu().float().numpy()
        audio = audio * 32768  # hps.data.max_wav_value
        audio = audio.astype(np.int16)
        return audio


def add_fades(audio, sample_rate, fade_length=0.25):
    fade_length = int(fade_length * sample_rate)
    fade = np.linspace(0, 1, fade_length)

    # Fade in
    audio[:fade_length] *= fade
    # Fade out
    audio[-fade_length:] *= fade[::-1]
    return audio


def get_song_data():
    fileid, name, phones, pitchid, dur, gtdur, slur = data[current_song].split("|")
    phones = phones.split(" ")
    pitchid = pitchid.split(" ")
    dur = [float(d) for d in dur.split(" ")]
    slur = [int(s) for s in slur.split(" ")]
    return phones, pitchid, dur, slur


def set_next_song():
    global current_song
    current_song += 1
    if current_song >= MAX_SONGS:
        current_song = 0
    return data[current_song].split("|")[1]


class LearnerStateModel:
    def __init__(self, init_motivation, init_artistry):
        self.motivation = init_motivation
        self.artistry = init_artistry
        self.std = 5
        self.max_mistake_memory = 10
        self.motivation_factor = 0.01
        self.mistakes_history_factor = 0.001
        self.mistakes_deque = deque(maxlen=self.max_mistake_memory)
        self.mistake_factor = self.motivation * self.motivation_factor
        self.mistake_memory = self.max_mistake_memory
        self.time_step = 0
        # Initial confidence is close to artistry
        self.confidence = np.clip(self.artistry - 10, 0, 100)
        # Initial mistakes is close to artistry
        self.mistakes_mean = np.clip(
            100 - self.artistry * 0.55 - self.motivation * 0.45, 0, 100
        )
        self.mistakes = int(
            np.clip(np.random.normal(self.mistakes_mean, self.std), 0, 100)
        )

    def get_confidence(self):
        return self.confidence

    def get_mistakes(self):
        return self.mistakes

    def get_motivation(self):
        return self.motivation

    def get_artistry(self):
        return self.artistry

    def get_time_step(self):
        return self.time_step

    def get_mistake_memory(self):
        return self.mistake_memory

    def step(self):
        # State Machine
        # Number of new mistakes = number of new mistakes + artistry + motivation * motivation factor + total_mistakes
        self.mistakes_mean = np.clip(
            100 - self.artistry * 0.55 - self.motivation * 0.45, 0, 100
        )
        self.mistakes = int(
            np.clip(np.random.normal(self.mistakes_mean, self.std), 0, 100)
        )
        self.mistakes_deque.appendleft(self.mistakes)
        # Confidence = number of new mistakes * mistake factor + artistry
        self.confidence_mean = self.artistry - self.mistakes * self.mistake_factor
        self.confidence = int(
            np.clip(np.random.normal(self.confidence_mean, self.std), 0, 100)
        )
        # Artistry = number of new mistakes
        self.artistry_mean = 100 - self.mistakes + self.time_step * self.motivation
        self.artistry = (
            self.artistry
            + int(np.clip(np.random.normal(self.artistry_mean, self.std), 0, 100))
        ) / 2
        # Motivation = artistry + total mistakes
        self.motivation_mean = (
            self.artistry
            - sum(list(self.mistakes_deque)[: self.mistake_memory])
            * self.mistakes_history_factor
        )
        self.motivation = (
            self.motivation
            + int(np.clip(np.random.normal(self.motivation_mean, self.std), 0, 100))
        ) / 2
        # Mistake factor = motivation * motivation factor
        self.mistake_factor = self.motivation * self.motivation_factor
        # Mistake memory = confidence
        self.mistake_memory = int(
            self.max_mistake_memory - self.confidence / self.max_mistake_memory
        )
        self.mistakes_history_factor = (100 - self.motivation / 100) * 0.01
        self.time_step += 1


class LoudnessNormalize(torch.nn.Module):
    def __init__(self, sample_rate: float, target_lufs_db: float = -32.0) -> None:
        super().__init__()
        self.meter = pyln.Meter(sample_rate)
        self.target_lufs_db = target_lufs_db

    def forward(self, x: torch.Tensor):
        x_lufs_db = self.meter.integrated_loudness(x.T)
        delta_lufs_db = np.array([self.target_lufs_db - x_lufs_db]).astype(np.float32)
        gain_lin = 10.0 ** (np.clip(delta_lufs_db, -120, 40.0) / 20.0)
        return gain_lin * x
