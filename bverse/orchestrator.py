import logging
import time
from queue import Queue, Empty
from typing import Any

import numpy as np
import sounddevice as sd

from bverse.config import Config
from bverse.dialogue.manager import DialogueManager
from bverse.audio.detector import AudioDetector
from bverse.audio.diarizer import AudioDiarizer
from bverse.speech.transcriber import SpeechTranscriber

logger = logging.getLogger(__name__)
logging.getLogger("faster_whisper").setLevel(logging.WARNING)


class Orchestrator:
    """
    Coordinates real-time audio processing and LLM-based dialogue management.

    This class captures live audio, detects speech activity, segments speech based on silence,
    transcribes the segmented audio, and uses a language model to generate context-aware insights.

    Key Responsibilities:
        - Capture audio in real-time via `sounddevice.InputStream`.
        - Segment audio based on speech activity and silence duration.
        - Transcribe segmented audio using `SpeechTranscriber`.
        - Pass transcriptions to `DialogueManager` to generate LLM-based responses.
        - Queue audio chunks for processing in the main thread (avoiding callback blocking).
    """

    def __init__(self, config: Config) -> None:
        """
        Initializes the Orchestrator with audio processing, transcription, and LLM configuration.

        Args:
            config: Configuration object containing audio processing, segmentation thresholds, and LLM model settings.
        """
        self.config = config

        # Core audio and language model components
        self.speech_transcriber = SpeechTranscriber()
        self.audio_diarizer = AudioDiarizer()
        self.audio_detector = AudioDetector()
        self.dialogue_manager = DialogueManager(config)

        # Internal state for buffering audio and tracking speech activity
        self.audio_buffer: list[np.ndarray] = []
        self.speech_duration = 0.
        self.silence_duration = 0.
        self.is_speaking = False

        # Queue for transferring audio chunks from callback to main thread
        self.chunk_queue: Queue[tuple[np.ndarray, str]] = Queue()

    def run(self) -> None:
        """
        Starts capturing and processing live audio.

        Audio is continuously captured and segmented in a callback thread.
        Segmented chunks are queued for transcription and LLM processing in the main thread.
        This method blocks until interrupted by the user.

        The process terminates gracefully when interrupted (e.g., with Ctrl+C).
        """
        logger.info("Starting Orchestrator. Press Ctrl+C to stop.")

        try:
            with sd.InputStream(
                samplerate=self.config.sample_rate,
                channels=1,
                dtype=np.float32,
                blocksize=self.config.chunk_size,
                callback=self._audio_callback,
            ):
                # Continuously process queued audio chunks in the main thread
                while True:
                    self._consume_queue()
                    time.sleep(0.1)
        except KeyboardInterrupt:
            logger.info("Orchestrator stopped by user.")

    def _audio_callback(
        self,
        indata: np.ndarray,
        _frames: int,
        _timestamp: Any,
        status: sd.CallbackFlags,
    ) -> None:
        """
        Handles incoming audio chunks from `sounddevice.InputStream` in the callback thread.

        Speech detection is performed, and chunks are queued for further processing when
        segmentation conditions are met.

        Args:
            indata: Captured audio chunk as a NumPy array.
            _frames: Number of frames in the current chunk (unused).
            _timestamp: Timestamp metadata from `sounddevice` (unused).
            status: Callback status flags indicating stream issues.
        """
        if status:
            logger.warning("Input stream status: %s", status)

        chunk = indata.flatten()
        self.audio_buffer.append(chunk)

        if self.audio_detector.is_speaking(chunk):
            self.is_speaking = True
            self.speech_duration += len(chunk) / self.config.sample_rate
            self.silence_duration = 0.
        else:
            self.silence_duration += len(chunk) / self.config.sample_rate

        # Evaluate segmentation criteria
        if self.speech_duration + self.silence_duration >= self.config.max_speech_sec:
            self._cut_chunk_and_queue("Maximum Speech Duration")
        elif self.is_speaking:
            if (
                    self.speech_duration >= self.config.min_speech_sec
                    and self.silence_duration >= self.config.short_silence_sec
            ):
                self._cut_chunk_and_queue("Natural Break After Minimum Speech")
            elif self.silence_duration >= self.config.long_silence_sec:
                self._cut_chunk_and_queue("Extended Silence")

    def _cut_chunk_and_queue(self, cut_reason: str) -> None:
        """
        Segments the current audio buffer, resets state, and queues the chunk for processing.

        Args:
            cut_reason: Reason the chunk was segmented (e.g., "Maximum Speech Duration").
        """
        if not self.audio_buffer:
            return

        audio_data = np.concatenate(self.audio_buffer)
        chunk_duration = len(audio_data) / self.config.sample_rate

        logger.info("Segmented audio chunk due to '%s' (Duration: %.2f s).", cut_reason, chunk_duration)

        self.chunk_queue.put((audio_data, cut_reason))

        # Reset state for the next segment
        self.audio_buffer = []
        self.speech_duration = 0.0
        self.silence_duration = 0.0
        self.is_speaking = False

    def _consume_queue(self) -> None:
        """
        Processes audio chunks from the queue in the main thread.

        Transcribes audio and generates LLM-based insights. This method is non-blocking.
        """
        try:
            while True:
                audio_data, cut_reason = self.chunk_queue.get_nowait()
                chunk_duration = len(audio_data) / self.config.sample_rate
                total_chunks = self.chunk_queue.qsize()
                logger.info("Processing audio chunk (Reason: '%s', Duration: %.2f s). Remaining chunks: %d.",
                            cut_reason, chunk_duration, total_chunks)
                self._process_chunk(audio_data, cut_reason)
        except Empty:
            pass

    def _process_chunk(self, audio_data: np.ndarray, cut_reason: str) -> None:
        """
        Transcribes an audio chunk and generates context-aware insights using dialogue manager.

        Args:
            audio_data: Audio data representing the segmented speech.
            cut_reason: Reason the chunk was segmented (e.g., "Maximum Speech Duration").
        """
        # TODO: Implement diarization for multi-speaker differentiation.
        # segments = self.audio_diarizer.diarize(audio_data)
        # ...

        transcription = self.speech_transcriber.transcribe(audio_data)

        if transcription:
            logger.info("Transcription: %s", transcription)

            self.dialogue_manager.add_utterance("Conversation", transcription)
            insight = self.dialogue_manager.suggest_response()

            logger.info("Assistant Insight: %s", insight)
        else:
            logger.info("No transcription produced for this chunk.")
