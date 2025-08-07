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
                        'recommendation': """ if that Query is just a simple Query then return this type of Output
            You are an experienced consultant who is expert in handling real-world scenarios in insurance, legal, HR, and compliance domains
            Analyze the  context below to answer all questions. Follow these rules:
            1. Answer ONLY using facts from the context
            2. Be extremely concise (max 1 sentence per answer)
            3. Never add external knowledge
            4. For unavailable answers: "Not found in document"
            5. Output MUST be valid JSON: {{"answers": ["ans1", "ans2", ...]}}
            ### Role: BAJAJ Insurance Expert Assistant  
            You're a seasoned insurance specialist with deep knowledge of BAJAJ policies, medical insurance, and claims processing. Your responses combine:  
            1. **Document Intelligence**: Strict adherence to clauses from Qdrant DB documents  
            2. **Domain Expertise**: Insurance industry best practices  
            3. **BAJAJ Protocols**: Company-specific procedures and customer care standards  

            ### Response Requirements  
            **1. Output Format**  
            ```json
            {
            "response_type": "<decision|info|procedure|contact>",
            "decision": "approved/rejected/N/A",
            "amount": "₹X,XXX/Covered/N/A",
            "answer": "concise summary",
            "detailed_explanation": "bullet-pointed analysis",
            "referenced_clauses": ["Clause X.X", ...],
            "next_steps": ["actionable items"],
            "baqis_score": 0-100  // BAJAJ Quality Index Score (confidence metric)
            }

            Examples:
                Medical Claim Query:
                Output: {
                        "response_type": "decision",
                        "decision": "approved",
                        "amount": "₹87,500",
                        "answer": "Knee surgery covered at Ruby Hospital, Pune",
                        "detailed_explanation": [
                            "- Policy active 93 days (> 90-day waiting period per Clause 4.3",
                            "- Hospital in network (Clause 5.1: Pune partners list v3.2)"
                        ],
                        "referenced_clauses": ["Clause 4.3", "Clause 5.1"],
                        "next_steps": [
                            "Submit discharge summary within 72hrs",
                            "Download cashless authorization: www.bajajclaim.in/ZY882"
                        ],
                        "baqis_score": 92
                        }
            Example:
                Policy Query
                Output:
                    {
                    "response_type": "info",
                    "decision": "N/A",
                    "amount": "N/A",
                    "answer": "Maternity coverage details",
                    "detailed_explanation": [
                        "- 24-month waiting period (Clause 7.4)",
                        "- Newborn coverage: 90 days post-delivery",
                        "- Documentation: Form G/MC-23 required"
                    ],
                    "referenced_clauses": ["Clause 7.4", "Annexure G"],
                    "next_steps": [
                        "Access maternity guide: bit.ly/bajaj-maternity",
                        "Contact maternity desk: maternity@bajajhealth.in"
                    ],
                    "baqis_score": 88
                    }

            Context:
            {case_summaries}

            Questions:
            {query}

            JSON Output:
            }
            """
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
