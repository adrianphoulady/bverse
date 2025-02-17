# bVerse

**bVerse** is a real-time conversational intelligence framework that transforms live audio streams into actionable insights. By integrating **speech detection**, **transcription**, **speaker diarization**, and **large language model (LLM) reasoning**, bVerse delivers contextually aware responses in multi-speaker environments. It serves as a bridge between human conversation and AI understanding, enhancing collaborative discussions, improving meeting productivity, and enabling interactive voice systems.

## Status
**Under Development** – The project is actively evolving. Its structure and features are subject to change.

## Key Features
- **Real-Time Audio Capture & Segmentation:** Detects speech activity and segments audio in real time.
- **Speech-to-Text Transcription:** Converts speech into text using state-of-the-art models.
- **Speaker Diarization:** Identifies and differentiates between speakers.
- **LLM-Powered Insights:** Generates contextually relevant responses based on ongoing conversation.
- **Token-Aware Context Management:** Optimizes conversation history handling within LLM token limits.

## Example Usage
```python
from bverse.config import Config
from bverse.orchestrator import Orchestrator

Orchestrator(Config()).run()
```

## Contributing
Contributions are welcome. Please submit issues and pull requests to the repository.

## License
This project is licensed under the MIT License.
