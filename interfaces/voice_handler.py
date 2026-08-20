import os
import time
import logging
from sarvamai import SarvamAI

logger = logging.getLogger(__name__)

class VoiceHandler:
    def __init__(self):
        # Initialize the client.
        self.client = SarvamAI(api_subscription_key=os.getenv("SARVAM_API_KEY"))

    def transcribe_and_translate(self, audio_file_path: str, max_retries: int = 3) -> str:
        """
        Takes an audio file path, sends it to Sarvam Saaras API, and returns translated English text.
        Includes basic retry logic for resilience.
        """
        for attempt in range(max_retries):
            try:
                # FIX: Explicitly open the file in binary read mode before passing it to the SDK
                with open(audio_file_path, "rb") as audio_file:
                    response = self.client.speech_to_text.transcribe(
                        file=audio_file,
                        model="saaras:v3",
                        mode="translate"
                    )
                return response.transcript
            except Exception as e:
                logger.warning(f"STT attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    raise RuntimeError(f"Speech-to-Text translation failed after {max_retries} attempts: {str(e)}")
                time.sleep(1)  # Simple backoff