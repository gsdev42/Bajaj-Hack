from openai import OpenAI
from dotenv import load_dotenv
import os
import logging
import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass
from legal_docs.models import RetrievalResult

# Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

# ✅ Create client
client = OpenAI()

@dataclass
class CaseDocument:
    id: str
    content: str
    vector: np.ndarray
    metadata: Dict[str, Any]

# @dataclass
# class RetrievalResult:
#     case: CaseDocument
#     score: float
#     rank: int = 0

@dataclass
class GeneratedInsight:
    content: str
    confidence: float
    supporting_case_ids: List[str]
    insight_type: str

class CaseInsightGenerator:
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.model_name = model_name

        if not os.getenv("OPENAI_API_KEY"):
            logger.error("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
            self.llm_initialized = False
        else:
            logger.info("OpenAI LLM initialized.")
            self.llm_initialized = True

        self.prompt_templates = {
            'analysis': """Analyze the following {num_cases} legal cases to answer the question: "{query}"

Cases:
{case_summaries}

Provide a concise legal analysis with references to case IDs like [C001].""",
            'recommendation': """Based on the following cases, generate 3-5 actionable recommendations regarding: "{query}"

Cases:
{case_summaries}

Mention relevant cases like [C002] wherever applicable."""
        }

    def generate(self, query: str, retrieved_cases: List[RetrievalResult], insight_type: str = "analysis") -> GeneratedInsight:
        if not self.llm_initialized:
            return GeneratedInsight(
                content="LLM not initialized",
                confidence=0.0,
                supporting_case_ids=[],
                insight_type=insight_type
            )

        case_summaries = self._prepare_case_summaries(retrieved_cases)

        prompt = self.prompt_templates.get(insight_type, self.prompt_templates['analysis']).format(
            query=query,
            case_summaries=case_summaries,
            num_cases=len(retrieved_cases)
        )

        try:
            logger.info("Sending prompt to OpenAI LLM...")
            response = client.chat.completions.create(
                model=self.model_name,  # ✅ should be "gpt-3.5-turbo"
                messages=[
                    {"role": "system", "content": "You are a legal document analysis assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            llm_output = response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI generation failed: {str(e)}")
            llm_output = "Insight generation failed."

        verification = self._verify_insight(llm_output, retrieved_cases)

        return GeneratedInsight(
            content=llm_output,
            confidence=verification['confidence'],
            supporting_case_ids=verification['supporting_ids'],
            insight_type=insight_type
        )

    def _prepare_case_summaries(self, cases: List[RetrievalResult]) -> str:
        return "\n".join(
            f"[{res.case.id}] {res.case.content[:200].strip()}..." for res in sorted(cases, key=lambda x: -x.score)
        )

    def _verify_insight(self, text: str, cases: List[RetrievalResult]) -> Dict:
        case_ids = [res.case.id for res in cases]
        supporting_ids = [cid for cid in case_ids if f"[{cid}]" in text]
        confidence = min(0.95, len(supporting_ids) / len(cases) * 1.5) if cases else 0.0

        return {
            'confidence': round(confidence, 2),
            'supporting_ids': supporting_ids
        }
