import os
import logging
from openai import OpenAI
from sarvamai import SarvamAI
from typing import Tuple

logger = logging.getLogger(__name__)

class GuardrailManager:
    """
    Hybrid Defense System:
    Layer 1 -> Nvidia NemoGuard-8B (Content Safety)
    Layer 3 -> Sarvam-105B (Hallucination Detection)
    """
    def __init__(self):
        # Sarvam Setup (For generation / hallucination checks)
        sarvam_key = os.getenv("SARVAM_API_KEY")
        if not sarvam_key:
            raise ValueError("SARVAM_API_KEY environment variable is missing.")
        self.sarvam_client = SarvamAI(api_subscription_key=sarvam_key)

        # Nvidia Setup (For fast input guarding)
        nvidia_key = os.getenv("NVIDIA_API_KEY")
        if not nvidia_key:
            raise ValueError("NVIDIA_API_KEY environment variable is missing.")
        
        self.nvidia_client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=nvidia_key
        )

    def is_input_safe(self, query: str) -> Tuple[bool, str]:
            """
            Layer 1: Pre-Retrieval Input Guard using Nvidia NemoGuard.
            Requires a strict system taxonomy prompt sent via chat completions.
            """
            nemoguard_prompt = f"""Task: Check if there is unsafe content in the user message in the below conversation in accordance with our safety policy with the below categories. Provide a 'safe' or 'unsafe' rating for the user message. If there is unsafe content, please also provide a list of violated categories.

            < BEGIN UNSAFE CONTENT CATEGORIES>
            S1: Violence.
            S2: Sexual.
            S3: Criminal Planning/Confessions.
            S4: Guns and Illegal Weapons.
            S5: Controlled/Regulated Substances.
            S6: Suicide and Self Harm.
            S7: Hate/Identity Hate.
            S8: PII/Privacy.
            S9: Harassment.
            S10: Threat.
            S11: Illegal Activity.
            < END UNSAFE CONTENT CATEGORIES>
            
            user: {query}
            
            Provide your safety assessment for the above conversation in the below output JSON format:
            {{"User Safety": safe or unsafe rating for the user message, "Safety Categories": a comma-separated list of applicable safety           categories from the provided taxonomy. Omit if all safe.}}
            Do not include anything other than the output JSON in your response.
            Output JSON:"""

            try:
                # Switch back to chat.completions, but pass the taxonomy prompt as the user message
                response = self.nvidia_client.chat.completions.create(
                    model="nvidia/llama-3.1-nemoguard-8b-content-safety",
                    messages=[{"role": "user", "content": nemoguard_prompt}],
                    temperature=0.0,
                    max_tokens=64,
                    top_p=1.0
                )
                
                # The model outputs a JSON string
                result = response.choices[0].message.content.strip().lower()
                
                # Check for the literal string "unsafe" inside the returned JSON payload
                if '"user safety": "unsafe"' in result or '"user safety":"unsafe"' in result:
                    return False, f"Query blocked by NemoGuard. Payload classified as unsafe. Raw output: {result}"
                
                return True, "Safe"
                
            except Exception as e:
                logger.error(f"Nvidia Input guardrail failed: {str(e)}")
                return False, "Security check unavailable. Request denied."
            
    def is_context_relevant(self, top_score: float, threshold: float = 0.0) -> bool:
        """
        Layer 2: Post-Retrieval Relevance Guard (Local).
        """
        return top_score >= threshold

    def is_output_grounded(self, context: str, answer: str) -> bool:
        """
        Layer 3: Post-Generation Hallucination Guard (Sarvam).
        """
        refusals = ["i do not know", "मुझे नहीं पता", "not found in the provided context"]
        if any(r in answer.lower() for r in refusals):
            return True

        eval_prompt = (
            "You are a strict exact-match verification system. Read the Context and the Answer. "
            "Does the Answer state ANY specific facts, figures, or claims that are NOT explicitly "
            "present in the Context? "
            "Respond strictly with 'GROUNDED' if all facts are in the context, or 'HALLUCINATED' if external information was added."
        )
        
        user_payload = f"Context:\n{context}\n\nAnswer:\n{answer}"
        
        try:
            response = self.sarvam_client.chat.completions(
                model="sarvam-105b",
                messages=[
                    {"role": "system", "content": eval_prompt},
                    {"role": "user", "content": user_payload}
                ],
                temperature=0.0,
                max_tokens=10,
                reasoning_effort=None
            )
            result = response.choices[0].message.content.strip().upper()
            return "HALLUCINATED" not in result
        except Exception as e:
            logger.error(f"Hallucination guardrail failed: {str(e)}")
            return False