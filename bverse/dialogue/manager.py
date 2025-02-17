import logging

from transformers import AutoModelForCausalLM, AutoTokenizer

from bverse.config import Config

logger = logging.getLogger(__name__)


class DialogueManager:
    """
    Manages multi-party conversation history and generates context-aware responses using an LLM.

    This class maintains a rolling conversation history while respecting token limits.
    It constructs prompts based on prior context and generates contextually relevant responses.
    """

    def __init__(self, config: Config) -> None:
        """
        Initializes the DialogueManager with the provided configuration.

        Args:
            config: Configuration object containing model identifiers and token-related limits.
        """
        self.config = config

        logger.info("Initializing DialogueManager with model: %s.", self.config.model_id)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_id, torch_dtype="auto", device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_id)

        self.system_prompt = (
            "You are an advanced AI assistant specializing in analyzing multi-party conversations. "
            "Transcriptions may be incomplete, fragmented, or contain errors. Your goal is to provide "
            "concise, accurate, and contextually relevant insights and professional assistance. "
            f"Limit responses to a maximum of {self.config.max_response_tokens} tokens."
        )

        # Track token counts for context management
        self.system_prompt_token_count = self._count_tokens(self.system_prompt)
        self.current_token_count = self.system_prompt_token_count

        # Conversation history represented as a list of tuples: (speaker, utterance, token_count)
        self.conversation_history: list[tuple[str, str, int]] = []

    def _count_tokens(self, text: str) -> int:
        """Calculates the number of tokens in the provided text using the tokenizer.

        Args:
            text: The input string to tokenize.

        Returns:
            The number of tokens in the text.
        """
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _manage_context_length(self) -> None:
        """
        Ensures that the conversation history fits within the model's context window.

        If the token limit is exceeded:
            - Removes the oldest messages until the token count is within limits.
            - If only one message remains, and it still exceeds the limit, truncates the message.
        """
        while self.current_token_count > self.config.max_context_tokens and len(self.conversation_history) > 1:
            speaker, _, token_count = self.conversation_history.pop(0)
            self.current_token_count -= token_count
            logger.debug(
                "Removed oldest message from %s (tokens: %d). Current token count: %d.",
                speaker, token_count, self.current_token_count
            )

        if self.conversation_history and self.current_token_count > self.config.max_context_tokens:
            speaker, text, _ = self.conversation_history[0]
            available_tokens = self.config.max_context_tokens - self.system_prompt_token_count
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            truncated_tokens = tokens[-available_tokens:]
            truncated_text = self.tokenizer.decode(truncated_tokens, skip_special_tokens=True)

            self.conversation_history[0] = (speaker, truncated_text, len(truncated_tokens))
            self.current_token_count = self.system_prompt_token_count + len(truncated_tokens)
            logger.debug(
                "Truncated message from %s to fit within token limit (tokens: %d). Current token count: %d.",
                speaker, len(truncated_tokens), self.current_token_count
            )

    def add_utterance(self, speaker: str, utterance: str) -> None:
        """
        Adds a speaker's utterance to the conversation history.

        If the last message was from the same speaker, the messages are merged into a single entry.
        After adding an utterance, the context length is adjusted if needed.

        Args:
            speaker: The identifier or label of the speaker (e.g., 'Speaker 1').
            utterance: The transcribed text of what the speaker said.
        """
        if self.conversation_history and self.conversation_history[-1][0] == speaker:
            last_speaker, last_text, last_count = self.conversation_history.pop()
            merged_text = f"{last_text} {utterance}"
            merged_token_count = self._count_tokens(merged_text)

            self.conversation_history.append((last_speaker, merged_text, merged_token_count))
            self.current_token_count += merged_token_count - last_count
        else:
            new_token_count = self._count_tokens(utterance)
            self.conversation_history.append((speaker, utterance, new_token_count))
            self.current_token_count += new_token_count

        logger.debug(
            "Added utterance by %s. Current token count: %d / %d.",
            speaker, self.current_token_count, self.config.max_context_tokens,
        )

        self._manage_context_length()

    def suggest_response(self) -> str:
        """
        Generates a context-aware response based on the conversation history.

        Returns:
            The generated response from the LLM as a string.
        """
        conversation_text = "\n".join(f"{s}: {t}" for s, t, _ in self.conversation_history)
        logger.debug("Suggesting response for: %s", conversation_text)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": conversation_text},
        ]

        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)

        output = self.model.generate(
            **inputs,
            max_new_tokens=self.config.max_response_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        generated_ids = output[:, inputs.input_ids.shape[1]:]
        response = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()

        logger.debug("Generated response: %s", response)
        return response
