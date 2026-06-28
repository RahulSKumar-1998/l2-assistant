"""Generation quality evaluation suite using LLM-as-judge.

Scores AI-generated recommendations on four dimensions:
Groundedness, Relevance, Actionability, and Accuracy (1-5 scale).
Includes quality gate thresholds for CI/CD pipelines.
"""

import json
from typing import Any, Optional

from pydantic import BaseModel, Field


# ── Evaluation Data Models ──────────────────────────────────────────────────


class GenerationEvalCase(BaseModel):
    """A single generation evaluation case."""
    query_id: str = Field(..., description="Unique query identifier")
    incident_text: str = Field(..., description="Input incident description")
    context_docs: list[str] = Field(
        default_factory=list,
        description="Retrieved context documents used for generation",
    )
    generated_output: str = Field(
        ...,
        description="AI-generated recommendation text",
    )
    reference_answer: Optional[str] = Field(
        default=None,
        description="Optional ground-truth reference answer",
    )
    category: str = Field(default="", description="Incident category")


class JudgeScore(BaseModel):
    """Scores from the LLM judge on a single generation."""
    groundedness: int = Field(
        ..., ge=1, le=5,
        description="Is the answer grounded in the provided context? (1=fabricated, 5=fully grounded)",
    )
    relevance: int = Field(
        ..., ge=1, le=5,
        description="Is the answer relevant to the incident? (1=irrelevant, 5=directly addresses issue)",
    )
    actionability: int = Field(
        ..., ge=1, le=5,
        description="Can an L2 engineer act on the steps? (1=vague, 5=specific and actionable)",
    )
    accuracy: int = Field(
        ..., ge=1, le=5,
        description="Is the root cause analysis technically accurate? (1=wrong, 5=correct)",
    )
    reasoning: str = Field(
        default="",
        description="Judge's reasoning for the scores",
    )

    @property
    def average(self) -> float:
        """Compute average score across all dimensions."""
        return (self.groundedness + self.relevance + self.actionability + self.accuracy) / 4.0


class GenerationEvalReport(BaseModel):
    """Complete generation evaluation report."""
    eval_type: str = Field(default="generation", description="Evaluation type")
    total_cases: int = Field(default=0, description="Total evaluation cases")
    avg_groundedness: float = Field(default=0.0, description="Average groundedness score")
    avg_relevance: float = Field(default=0.0, description="Average relevance score")
    avg_actionability: float = Field(default=0.0, description="Average actionability score")
    avg_accuracy: float = Field(default=0.0, description="Average accuracy score")
    avg_overall: float = Field(default=0.0, description="Average overall score")
    individual_scores: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Per-case scores with query_id",
    )
    quality_gate_passed: bool = Field(
        default=False,
        description="Whether quality gate thresholds are met",
    )
    quality_gate_details: dict[str, Any] = Field(
        default_factory=dict,
        description="Details of quality gate evaluation",
    )


# ── Quality Gate Thresholds ─────────────────────────────────────────────────


QUALITY_THRESHOLDS = {
    "groundedness": 3.5,
    "relevance": 3.5,
    "actionability": 3.0,
    "accuracy": 3.5,
    "overall": 3.5,
}


# ── Judge Prompt Template ───────────────────────────────────────────────────


JUDGE_PROMPT_TEMPLATE = """You are an expert evaluator for an AI-powered IT incident support system.
Your task is to evaluate the quality of an AI-generated recommendation for an IT incident.

## Incident Description
{incident_text}

## Retrieved Context Documents
{context_docs}

## AI-Generated Recommendation
{generated_output}

{reference_section}

## Evaluation Criteria

Score each dimension from 1 (worst) to 5 (best):

1. **Groundedness** (1-5): Is the recommendation grounded in the provided context documents?
   - 1: Completely fabricated, no basis in context
   - 3: Partially grounded, some claims unsupported
   - 5: Fully grounded, all claims traceable to context

2. **Relevance** (1-5): Does the recommendation address the specific incident?
   - 1: Completely irrelevant to the incident
   - 3: Somewhat relevant but misses key aspects
   - 5: Directly and comprehensively addresses the incident

3. **Actionability** (1-5): Can an L2 support engineer act on the triage steps?
   - 1: Vague, no concrete actions
   - 3: Some actionable steps but lacks specifics
   - 5: Clear, specific, step-by-step actions with commands

4. **Accuracy** (1-5): Is the root cause analysis technically accurate?
   - 1: Technically incorrect
   - 3: Partially correct with some errors
   - 5: Technically sound and correct

## Response Format

Respond with ONLY a JSON object (no markdown, no code fences):
{{
    "groundedness": <1-5>,
    "relevance": <1-5>,
    "actionability": <1-5>,
    "accuracy": <1-5>,
    "reasoning": "<brief explanation of your scores>"
}}"""


# ── Generation Evaluator ───────────────────────────────────────────────────


class GenerationEvaluator:
    """Evaluates generation quality using an LLM-as-judge approach.

    The evaluator sends each generated recommendation to a judge LLM
    that scores it on four dimensions. Supports quality gate enforcement
    for CI/CD pipelines.

    Usage:
        evaluator = GenerationEvaluator(judge_llm_client)
        report = await evaluator.evaluate(eval_cases)
    """

    def __init__(
        self,
        judge_llm_client: Any,
        thresholds: Optional[dict[str, float]] = None,
    ) -> None:
        """Initialize the generation evaluator.

        Args:
            judge_llm_client: LLM client for the judge model. Must have an
                              async `generate(prompt: str) -> LLMResponse` method.
            thresholds: Optional custom quality thresholds. Defaults to
                       QUALITY_THRESHOLDS.
        """
        self.judge = judge_llm_client
        self.thresholds = thresholds or QUALITY_THRESHOLDS

    def build_judge_prompt(self, case: GenerationEvalCase) -> str:
        """Build the judge prompt for a single evaluation case.

        Args:
            case: Generation evaluation case with incident, context, and output.

        Returns:
            Formatted judge prompt string.
        """
        context_text = "\n---\n".join(case.context_docs) if case.context_docs else "(no context provided)"

        reference_section = ""
        if case.reference_answer:
            reference_section = f"""## Reference Answer (Ground Truth)
{case.reference_answer}
"""

        return JUDGE_PROMPT_TEMPLATE.format(
            incident_text=case.incident_text,
            context_docs=context_text,
            generated_output=case.generated_output,
            reference_section=reference_section,
        )

    async def judge_single(self, case: GenerationEvalCase) -> JudgeScore:
        """Score a single generation using the LLM judge.

        Args:
            case: Evaluation case to judge.

        Returns:
            JudgeScore with scores on all four dimensions.
        """
        prompt = self.build_judge_prompt(case)
        response = await self.judge.generate(prompt=prompt)

        # Parse JSON response from judge
        try:
            # Handle potential markdown code fences in response
            content = response.content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()

            scores = json.loads(content)
            return JudgeScore(**scores)
        except (json.JSONDecodeError, ValueError) as e:
            # Fallback: return neutral scores on parse failure
            return JudgeScore(
                groundedness=3,
                relevance=3,
                actionability=3,
                accuracy=3,
                reasoning=f"Judge response parse error: {str(e)[:100]}. Raw: {response.content[:200]}",
            )

    async def evaluate(
        self,
        eval_cases: list[GenerationEvalCase],
    ) -> GenerationEvalReport:
        """Run evaluation across all cases and produce a report.

        Args:
            eval_cases: List of generation evaluation cases.

        Returns:
            GenerationEvalReport with aggregate scores and quality gate status.
        """
        all_scores: list[JudgeScore] = []
        individual: list[dict[str, Any]] = []

        for case in eval_cases:
            score = await self.judge_single(case)
            all_scores.append(score)
            individual.append({
                "query_id": case.query_id,
                "category": case.category,
                "groundedness": score.groundedness,
                "relevance": score.relevance,
                "actionability": score.actionability,
                "accuracy": score.accuracy,
                "average": score.average,
                "reasoning": score.reasoning,
            })

        if not all_scores:
            return GenerationEvalReport(total_cases=0)

        n = len(all_scores)
        avg_g = sum(s.groundedness for s in all_scores) / n
        avg_r = sum(s.relevance for s in all_scores) / n
        avg_a = sum(s.actionability for s in all_scores) / n
        avg_ac = sum(s.accuracy for s in all_scores) / n
        avg_overall = (avg_g + avg_r + avg_a + avg_ac) / 4.0

        # Quality gate evaluation
        gate_details = {
            "groundedness": {"threshold": self.thresholds["groundedness"], "actual": avg_g, "passed": avg_g >= self.thresholds["groundedness"]},
            "relevance": {"threshold": self.thresholds["relevance"], "actual": avg_r, "passed": avg_r >= self.thresholds["relevance"]},
            "actionability": {"threshold": self.thresholds["actionability"], "actual": avg_a, "passed": avg_a >= self.thresholds["actionability"]},
            "accuracy": {"threshold": self.thresholds["accuracy"], "actual": avg_ac, "passed": avg_ac >= self.thresholds["accuracy"]},
            "overall": {"threshold": self.thresholds["overall"], "actual": avg_overall, "passed": avg_overall >= self.thresholds["overall"]},
        }

        all_passed = all(d["passed"] for d in gate_details.values())

        return GenerationEvalReport(
            eval_type="generation",
            total_cases=n,
            avg_groundedness=round(avg_g, 3),
            avg_relevance=round(avg_r, 3),
            avg_actionability=round(avg_a, 3),
            avg_accuracy=round(avg_ac, 3),
            avg_overall=round(avg_overall, 3),
            individual_scores=individual,
            quality_gate_passed=all_passed,
            quality_gate_details=gate_details,
        )
