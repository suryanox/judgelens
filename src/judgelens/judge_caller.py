import json
import re

class JudgeCaller:
    """
    Thin wrapper around litellm.completion.

    Parameters
    ----------
    judge_model     : any model string litellm understands
    prompt_template : must contain {input}, {response}, {ground_truth}
    temperature     : default 0 for determinism
    """

    def __init__(
        self,
        judge_model: str,
        prompt_template: str,
        temperature: float = 0.0,
    ):
        self.judge_model = judge_model
        self.prompt_template = prompt_template
        self.temperature = temperature

        try:
            import litellm
        except ImportError:
            raise ImportError(
                'litellm is required.\n'
                'Install with:  pip install "litellm<=1.70.0"'
            )

    def score(
        self,
        input: str,
        response: str,
        ground_truth: str,
    ) -> tuple[float, str]:
        """
        Call the judge and return (normalised_score: float 0–1, reasoning: str).

        Raises
        ------
        RuntimeError  if the judge returns unparseable output
        """
        prompt = self.prompt_template.format(
            input=input,
            response=response,
            ground_truth=ground_truth,
        )
        raw = self._call(prompt)
        return self._parse(raw)

    def with_prompt(self, new_template: str) -> "JudgeCaller":
        """Return a copy of this caller using a different prompt template."""
        return JudgeCaller(
            judge_model=self.judge_model,
            prompt_template=new_template,
            temperature=self.temperature,
        )

    def _call(self, prompt: str) -> str:
        import litellm

        kwargs: dict = {
            "model": self.judge_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": 256,
        }


        resp = litellm.completion(**kwargs)
        return resp.choices[0].message.content.strip()

    def _parse(self, raw: str) -> tuple[float, str]:
        """
        Parse judge output into (score 0–1, reasoning).

        Strategy
        --------
        1. Strip markdown fences the model may have wrapped around JSON.
        2. Try strict json.loads.
        3. Fall back to regex extraction for partial/malformed output.
        """
        cleaned = re.sub(
            r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.DOTALL
        )

        try:
            data = json.loads(cleaned)
            score = float(data["score"]) / 10.0
            reasoning = str(data.get("reasoning", "")).strip()
            return _clamp(score), reasoning
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            pass

        m = re.search(r'"score"\s*:\s*(\d+(?:\.\d+)?)', cleaned)
        if m:
            score = float(m.group(1)) / 10.0
            rm = re.search(r'"reasoning"\s*:\s*"([^"]*)"', cleaned)
            reasoning = rm.group(1).strip() if rm else raw[:200]
            return _clamp(score), reasoning

        raise RuntimeError(
            f"Judge ({self.judge_model}) returned unparseable output:\n{raw[:500]}"
        )

def _clamp(v: float) -> float:
    return min(max(v, 0.0), 1.0)
