import os

import numpy as np
import torch
from pyannote.audio import Pipeline
from pyannote.core import Annotation


class AudioDiarizer:
    """Performs speaker diarization using a pretrained Pyannote pipeline."""

    def __init__(self, model_id: str = "pyannote/speaker-diarization-3.1") -> None:
        """Initializes the AudioDiarizer with a pretrained diarization model.

        Args:
            model_id: The Hugging Face model identifier for the Pyannote diarization model.
        """
        auth_token = os.getenv("HF_ACCESS_TOKEN")
        self.pipeline = Pipeline.from_pretrained(model_id, use_auth_token=auth_token)

    def diarize(self, audio: np.ndarray, sample_rate: int = 16000) -> Annotation:
        """Performs speaker diarization on a raw audio waveform.

        Args:
            audio: A 1D NumPy array representing a mono-channel audio waveform.
            sample_rate: The sampling rate of the audio in Hz.

        Returns:
            A Pyannote Annotation object representing detected speaker segments.
        """
        waveform = torch.as_tensor(audio, dtype=torch.float32).unsqueeze(0)
        return self.pipeline({"waveform": waveform, "sample_rate": sample_rate})
