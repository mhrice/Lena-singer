from neutone_sdk import NeutoneParameter, WaveformToWaveformBase
from neutone_sdk.utils import save_neutone_model

import utils.utils as utils
import os
from egs.visinger2.models import SynthesizerTrn
from typing import Tuple, List, Dict, Optional
import argparse
import torch
import numpy as np
from text import npu
from torch import Tensor
from pathlib import Path
from torch.nn.utils.weight_norm import WeightNorm


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


class VISingerWrapper(WaveformToWaveformBase):
    def get_neutone_parameters(self) -> List[NeutoneParameter]:
        return [
            NeutoneParameter("min", "min clip threshold", default_value=0.15),
            NeutoneParameter("max", "max clip threshold", default_value=0.15),
            NeutoneParameter("gain", "scale clip threshold", default_value=1.0),
        ]

    @torch.jit.export
    def is_input_mono(self) -> bool:
        return True

    @torch.jit.export
    def is_output_mono(self) -> bool:
        return True

    @torch.jit.export
    def get_native_sample_rates(self) -> List[int]:
        return []  # Supports all sample rates

    @torch.jit.export
    def get_native_buffer_sizes(self) -> List[int]:
        return []  # Supports all buffer sizes

    def get_model_authors(self) -> List[str]:
        return ["Matthew Rice"]

    def get_model_name(self) -> str:
        return "VISinger"

    def get_model_short_description(self) -> str:
        return "Audio clipper."

    def get_model_long_description(self) -> str:
        return "Clips the input audio between -1 and 1."

    def get_technical_description(self) -> str:
        return "Clips the input audio between -1 and 1."

    def is_experimental(self) -> bool:
        return True

    def get_model_version(self) -> str:
        return "1.0.0"

    def get_tags(self) -> List[str]:
        return ["clipper"]

    @torch.no_grad()
    def do_forward_pass(self, x: Tensor, params: Dict[str, Tensor]) -> Tensor:
        min_val, max_val, gain = params["min"], params["max"], params["gain"]
        pho = "y v"
        pitchid = "C4 C4"
        dur = "0.232140 0.232140"
        slur = "0 0"
        gtdur = "0 0"

        pho, pitchid, dur, slur, gtdur = self.parse_label(
            pho, pitchid, dur, slur, gtdur
        )
        pho_lengths = torch.tensor([pho.size(0)])
        pho = pho.unsqueeze(0)
        pitchid = pitchid.unsqueeze(0)
        dur = dur.unsqueeze(0)
        slur = slur.unsqueeze(0)
        # return torch.randn(x.shape)
        o = torch.tensor([])
        o, _, _ = self.model.infer(pho, pho_lengths, pitchid, dur, slur)
        audio = o[0, 0].data.cpu().float()
        # x = self.model.forward(x, min_val, max_val, gain)
        return audio.unsqueeze(0)

    def parse_label(
        self,
        pho: str,
        pitchid: str,
        dur: str,
        slur: str,
        gtdur: str,
    ):
        phos: List[int] = []
        pitchs: List[int] = []
        durs: List[float] = []
        slurs: List[int] = []
        gtdurs: List[float] = []

        for index in range(len(pho.split())):
            phos.append(self.model.phone_to_int[pho.strip().split()[index]])
            pitchs.append(self.model.pitch_to_int[pitchid.strip().split()[index]])
            durs.append(float(dur.strip().split()[index]))
            slurs.append(int(slur.strip().split()[index]))
            gtdurs.append(float(gtdur.strip().split()[index]))

        # phos = np.asarray(phos, dtype=np.int32)
        # pitchs = np.asarray(pitchs, dtype=np.int32)
        # durs = np.asarray(durs, dtype=np.float32)
        # slurs = np.asarray(slurs, dtype=np.int32)
        # gtdurs = np.asarray(gtdurs, dtype=np.float32)
        # gtdurs = np.ceil(gtdurs / (hop_size / sample_rate))

        phos = torch.tensor(phos)
        pitchs = torch.tensor(pitchs)
        durs = torch.tensor(durs)
        slurs = torch.tensor(slurs)
        gtdurs = torch.tensor(gtdurs)
        return phos, pitchs, durs, slurs, gtdurs


def remove_weight_norm(module):
    module_list = [mod for mod in module.children()]
    if len(module_list) == 0:
        for k, hook in module._forward_pre_hooks.items():
            if isinstance(hook, WeightNorm):
                hook.remove(module)
                del module._forward_pre_hooks[k]
    else:
        for mod in module_list:
            remove_weight_norm(mod)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-model_dir", "--model_dir", type=str, required=True)
    args = parser.parse_args()

    model_dir = args.model_dir

    model, hps = load_model(model_dir)
    remove_weight_norm(model)
    # for layer in model.modules():
    #     if hasattr(layer, "remove_weight_norm"):
    #         try:
    #             layer.remove_weight_norm()
    #         except ValueError as e:
    #             print(e)
    model.phone_to_int = npu.symbol_converter.ttsing_phone_to_int
    model.pitch_to_int = npu.symbol_converter.ttsing_opencpop_pitch_to_int

    wrapper = VISingerWrapper(model)
    in_noise = torch.randn(1, 11264)

    root_dir = Path("./neutone_out")
    # wrapper(in_noise)
    save_neutone_model(
        wrapper, root_dir, freeze=True, dump_samples=True, submission=True
    )
