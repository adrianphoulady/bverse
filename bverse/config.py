from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    """
    Configuration settings for Verse, including model and audio processing parameters.

    Attributes:
        model_id: Hugging Face model identifier for the LLM.
        max_context_tokens: Maximum number of tokens allowed in the conversation context.
        max_response_tokens: Maximum number of tokens allowed in a generated response.

        sample_rate: Audio sampling rate in Hz.
        chunk_size: Number of frames per audio chunk.

        min_speech_sec: Minimum duration of speech in seconds required before processing.
        short_silence_sec: Duration of short silence in seconds that triggers chunk processing after minimum speech.
        long_silence_sec: Duration of long silence in seconds that triggers chunk processing.
        max_speech_sec: Maximum duration of uninterrupted speech in seconds before forcing processing.
    """

    # Model and token configurations
    model_id: str = "Qwen/Qwen2.5-7B-Instruct"
    max_context_tokens: int = 4096
    max_response_tokens: int = 128

    # Audio parameters
    sample_rate: int = 16000
    chunk_size: int = 512

    # Speech-silence segmentation thresholds in seconds
    min_speech_sec: float = 10.0
    short_silence_sec: float = 0.5
    long_silence_sec: float = 2.0
    max_speech_sec: float = 30.0
