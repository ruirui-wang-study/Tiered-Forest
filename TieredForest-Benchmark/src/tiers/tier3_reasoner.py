import re
import time
from typing import List, Optional

from openai import OpenAI

from .. import config
from ..utils.cache_manager import LLMCache
from ..utils.retry_handler import retry_on_failure


class LLMReasoner:
    """
    Tier 3 LLM reasoning with optional constrained decoding over candidate choices.
    """

    def __init__(self, cache_manager: LLMCache):
        self.cache = cache_manager
        self.client = OpenAI(
            api_key=config.DEEPSEEK_API_KEY,
            base_url=config.DEEPSEEK_BASE_URL,
        )
        self.model_name = config.MODEL_NAME

    @retry_on_failure(max_retries=3, backoff_base=2.0)
    def reason(
        self,
        question: str,
        context: Optional[str] = None,
        choices: Optional[List[str]] = None,
        max_tokens: int = 64,
    ) -> tuple:
        """
        Returns (answer, prompt_tokens, completion_tokens, latency)
        """
        prompt = self._build_prompt(question, context=context, choices=choices)

        cached = self.cache.get(prompt, self.model_name)
        if cached:
            answer = self._postprocess_answer(cached)
            if choices:
                answer = self._align_to_choices(answer, choices)
            return answer, 0, 0, 0.0

        start_time = time.time()
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.0,
                stream=False,
            )
            latency = time.time() - start_time

            answer = self._postprocess_answer(response.choices[0].message.content)
            if choices:
                answer = self._align_to_choices(answer, choices)

            usage = response.usage
            prompt_tokens = usage.prompt_tokens
            completion_tokens = usage.completion_tokens

            self.cache.set(prompt, self.model_name, answer)
            return answer, prompt_tokens, completion_tokens, latency
        except Exception as e:
            print(f"LLM API Error: {e}")
            raise

    def reason_simple(self, question: str) -> tuple:
        prompt = (
            "Answer this factual question with only the final short phrase.\n"
            f"Question: {question}\n"
            "Answer:"
        )

        cached_response = self.cache.get(prompt, self.model_name)
        if cached_response:
            return self._postprocess_answer(cached_response), 0, 0, 0.0

        start_time = time.time()
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.0,
                stream=False,
            )
            latency = time.time() - start_time

            answer = self._postprocess_answer(response.choices[0].message.content)
            usage = response.usage

            self.cache.set(prompt, self.model_name, answer)
            return answer, usage.prompt_tokens, usage.completion_tokens, latency
        except Exception as e:
            print(f"LLM API Error: {e}")
            raise

    def _build_prompt(
        self,
        question: str,
        context: Optional[str] = None,
        choices: Optional[List[str]] = None,
    ) -> str:
        if choices:
            options = [c.strip() for c in choices if c and c.strip()]
            option_lines = "\n".join(f"{i+1}. {c}" for i, c in enumerate(options))
            prompt = (
                "You answer WebQSP-style factual questions.\n"
                "You MUST return answers using only exact text copied from Candidate Options.\n"
                "Return one or more options separated by commas; no explanation.\n\n"
                f"Question: {question}\n"
            )
            if context:
                prompt += f"Context hint: {context}\n"
            prompt += f"Candidate Options:\n{option_lines}\nAnswer:"
            return prompt

        if context:
            return (
                "You answer WebQSP-style factual questions.\n"
                "Use the context as a hint only; if context conflicts with known facts, ignore it.\n"
                "Return ONLY the shortest final answer phrase (entity/date/number), no explanation.\n"
                "If there are multiple answers, separate them with commas.\n\n"
                f"Context hint: {context}\n"
                f"Question: {question}\n"
                "Answer:"
            )

        return (
            "You answer WebQSP-style factual questions.\n"
            "Return ONLY the shortest final answer phrase (entity/date/number), no explanation.\n"
            "If there are multiple answers, separate them with commas.\n\n"
            f"Question: {question}\n"
            "Answer:"
        )

    def _postprocess_answer(self, text: str) -> str:
        answer = "" if text is None else str(text).strip()
        if not answer:
            return answer

        match = re.search(r"\{([^{}]+)\}", answer)
        if match:
            answer = match.group(1).strip()

        prefixes = ("answer:", "final answer:")
        lowered = answer.lower()
        for prefix in prefixes:
            if lowered.startswith(prefix):
                answer = answer[len(prefix) :].strip()
                break

        answer = answer.splitlines()[0].strip()
        return answer.strip("`\"' ")

    def _normalize_for_match(self, text: str) -> str:
        return re.sub(r"[\W_]+", "", text.lower())

    def _align_to_choices(self, answer: str, choices: List[str]) -> str:
        normalized_choices: List[str] = []
        seen = set()
        for choice in choices:
            c = choice.strip()
            if not c:
                continue
            key = c.lower()
            if key in seen:
                continue
            seen.add(key)
            normalized_choices.append(c)

        if not normalized_choices:
            return answer

        chunks = [part.strip() for part in re.split(r"[,;\n]+", answer) if part.strip()]
        if not chunks:
            chunks = [answer.strip()]

        selected: List[str] = []
        for chunk in chunks:
            best = self._best_choice(chunk, normalized_choices)
            if best and best not in selected:
                selected.append(best)

        if selected:
            return ", ".join(selected[:3])

        fallback = self._best_choice(answer, normalized_choices)
        return fallback if fallback else answer

    def _best_choice(self, text: str, choices: List[str]) -> Optional[str]:
        if not text:
            return None

        raw = text.strip()
        norm = self._normalize_for_match(raw)
        if not norm:
            return None

        for choice in choices:
            if raw.lower() == choice.lower():
                return choice

        for choice in choices:
            c_norm = self._normalize_for_match(choice)
            if norm == c_norm:
                return choice
            if norm in c_norm or c_norm in norm:
                return choice

        text_tokens = set(re.findall(r"[a-z0-9]+", raw.lower()))
        if not text_tokens:
            return None

        best_choice = None
        best_score = 0.0
        for choice in choices:
            choice_tokens = set(re.findall(r"[a-z0-9]+", choice.lower()))
            if not choice_tokens:
                continue
            inter = len(text_tokens & choice_tokens)
            union = len(text_tokens | choice_tokens)
            score = inter / union if union else 0.0
            if score > best_score:
                best_score = score
                best_choice = choice

        return best_choice if best_score >= 0.25 else None
