import numpy as np
from silero_vad import VADIterator, load_silero_vad


class AudioDetector:
    """Detects speech activity in audio chunks using the Silero VAD model."""

    def __init__(self, threshold: float = 0.5, sampling_rate: int = 16000, onnx: bool = True) -> None:
        """Initializes the AudioDetector with Silero VAD model.

        Args:
            threshold: Voice activity detection (VAD) threshold (between 0 and 1).
            sampling_rate: Sampling rate of the audio input.
            onnx: Whether to load the ONNX version of the Silero VAD model.
        """
        self.model = load_silero_vad(onnx=onnx)
        self.vad = VADIterator(self.model, threshold=threshold, sampling_rate=sampling_rate)

    def is_speaking(self, audio_chunk: np.ndarray) -> bool:
        """
        Determines if the given audio chunk contains speech.

        Args:
            audio_chunk: A 1D NumPy array representing a mono audio segment.

        Returns:
            True if speech is detected, False otherwise.
        """
        self.vad(audio_chunk)
        return self.vad.triggered
