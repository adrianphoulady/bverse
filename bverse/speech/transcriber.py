from faster_whisper import WhisperModel
import numpy as np


class SpeechTranscriber:
    """
    Handles speech-to-text transcription using the Faster-Whisper model.
    """

    def __init__(self, model_size: str = "large-v3-turbo") -> None:
        """
        Initializes the transcriber with the specified Whisper model.

        Args:
            model_size: The size or variant of the Whisper model to load.
        """
        self.model = WhisperModel(model_size, device="auto", compute_type="auto")

    def transcribe(self, audio_segment: np.ndarray) -> str | None:
        """
        Transcribes an audio segment into text.

        Args:
            audio_segment: A NumPy array representing the audio data (1D, mono).

        Returns:
            The transcribed text if speech is detected; otherwise, None.
        """
        segments, _ = self.model.transcribe(audio_segment, language="en")
        text = " ".join(seg.text for seg in segments)

        return text or None
