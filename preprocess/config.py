
class NormType:
    RAW = "raw"
    MINMAX = "minmax"
    ZSCORE = "z-score"

class MelABSValue:
    TTS = 1.0
    VOCODER = 1.0
    VC = 4.0


class BasicConfig:
    def __init__(
            self,
            sample_rate,
            n_fft,
            hop_size,
            win_size,
            fmin,
            fmax,
            rescale = False,
            trim_silence = False,
            save_align_wavs = False,
            lf0_inter = False,
            lf0_norm = NormType.RAW,
            energy_norm = NormType.RAW,
            min_level_db = -115,
            ref_level_db = 20,
            max_abs_value = MelABSValue.TTS,
            min_db = -115,
            signal_normalization = True,
            allow_clipping_in_normalization = True,
            symmetric_acoustic = True,        
        ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.rescale = rescale
        self.trim_silence = trim_silence
        self.save_align_wavs = save_align_wavs
        self.lf0_inter = lf0_inter
        self.lf0_norm = lf0_norm
        self.energy_norm = energy_norm
        self.min_level_db = min_level_db
        self.ref_level_db = ref_level_db
        self.signal_normalization = signal_normalization
        self.allow_clipping_in_normalization = allow_clipping_in_normalization
        self.symmetric_acoustic = symmetric_acoustic
        self.max_abs_value = max_abs_value
        self.min_db = min_db

    def __repr__(self) -> str:
        content = \
        f"sample_rate: {self.sample_rate}\n" + \
        f"n_fft: {self.n_fft}\n" + \
        f"hop_size: {self.hop_size}\n" + \
        f"win_size: {self.win_size}\n" + \
        f"fmin: {self.fmin}\n" + \
        f"fmax: {self.fmax}\n" + \
        f"rescale,: {self.rescale}\n" + \
        f"trim_silence: {self.trim_silence}\n" + \
        f"save_align_wavs: {self.save_align_wavs}\n" + \
        f"lf0_inter: {self.lf0_inter}\n" + \
        f"lf0_norm: {self.lf0_norm}\n" + \
        f"energy_norm: {self.energy_norm}\n" + \
        f"min_level_db: {self.min_level_db}\n" + \
        f"ref_level_db: {self.ref_level_db}\n" + \
        f"signal_normalization: {self.signal_normalization}\n" + \
        f"allow_clipping_in_normalization: {self.allow_clipping_in_normalization}\n" + \
        f"symmetric_acoustic: {self.symmetric_acoustic}\n" + \
        f"max_abs_value: {self.max_abs_value}\n" + \
        f"min_db: {self.min_db}\n"
        return content

preConfiged48K = BasicConfig(
    sample_rate = 48000, 
    n_fft = 4096, 
    hop_size = 600, 
    win_size = 2400, 
    fmin = 0, 
    fmax = 24000
)

preConfiged44K = BasicConfig(
    sample_rate = 44100, 
    n_fft = 2048, 
    hop_size = 512, 
    win_size = 2048, 
    fmin = 0, 
    fmax = 22050
)

preconfiged24K256 = BasicConfig(
    sample_rate = 24000, 
    n_fft = 1024, 
    hop_size = 256, 
    win_size = 1024, 
    fmin = 0, 
    fmax = 12000
)

preConfiged24K = BasicConfig(
    sample_rate = 24000, 
    n_fft = 2048, 
    hop_size = 300, 
    win_size = 1200, 
    fmin = 0, 
    fmax = 12000
)

preConfiged22K = BasicConfig(
    sample_rate = 22050, 
    n_fft = 1024, 
    hop_size = 256, 
    win_size = 1024, 
    fmin = 0, 
    fmax = 11025
)

preConfiged16K = BasicConfig(
    sample_rate = 16000, 
    n_fft = 1024, 
    hop_size = 200, 
    win_size = 800, 
    fmin = 0, 
    fmax = 8000
)

preConfiged16K_10ms = BasicConfig(
    sample_rate = 16000, 
    n_fft = 1024, 
    hop_size = 160, 
    win_size = 640, 
    fmin = 0, 
    fmax = 8000
)
