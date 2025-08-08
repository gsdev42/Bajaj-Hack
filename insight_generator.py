import json
import logging
from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

client = OpenAI(timeout=60.0)  # Added timeout for Render deployment

@dataclass
class RetrievalResult:
    case: Any  # Should reference your CaseDocument type
    score: float

@dataclass
class GeneratedInsight:
    content: Any  # Can be str or dict
    confidence: float
    supporting_case_ids: List[str]
    insight_type: str

class CaseInsightGenerator:
    def __init__(self, model_name: str = "gpt-3.5-turbo-0125"):  # Faster and cheaper model
        self.model_name = model_name
        self.llm_initialized = bool(os.getenv("OPENAI_API_KEY"))
        self.token_counts = []  # For monitoring
        
self.prompt_templates = {
    'analysis': """**Legal Analysis Request**: {query}
    
    **Relevant Cases** (Top 3):
    {case_summaries}
    
    **Instructions**:
    1. Provide concise analysis (max 100 words)
    2. Reference case IDs like [C001]
    3. Use ONLY case content
    4. For missing information: State "Not explicitly stated" and make logical inferences using insurance domain knowledge""",
    
    'recommendation': """**Role**: Senior Insurance/Legal Consultant at BAJAJ
    **Task**: Generate comprehensive JSON response for: "{query}"
    
    **Policy Context** (Top 3 documents):
    {case_summaries}
    
    **Response Strategy**:
    1. FIRST: Use exact matches from context
    2. SECOND: If not found, make logical inferences based on:
        - Standard insurance industry practices
        - BAJAJ policy patterns
        - Regulatory guidelines (IRDAI)
    3. CLEARLY INDICATE inferred content with "(inferred)"
    4. For completely unknown information: Use "Not specified"
    
    **Output Format** (Strict JSON):
    {{
        "response_type": "decision|info|procedure|contact",
        "decision": "approved/rejected/Not specified",
        "amount": "₹.../Covered/Not specified",
        "answer": "1-sentence summary",
        "detailed_explanation": [
            "bullet 1 [source: Context/Inference]",
            "bullet 2 [source: Context/Inference]"
        ],
        "referenced_clauses": ["Clause X.X"],
        "next_steps": ["actionable items"],
        "confidence_source": "Exact match/Partial match/Inference"
    }}
    
    **Rules**:
    1. MAX 3 bullet points in explanations
    2. Tag sources: [Context] or [Inference]
    3. confidence_source values:
        - "Exact match": All info from context
        - "Partial match": Mix of context and inference
        - "Inference": All info logically extrapolated
    4. Never fabricate specific numbers or clauses
    
    **Inference Examples**:
    1. Context: "Grace period mentioned but not specified"
       Acceptable: "Standard 30-day grace period (inferred)"
    
    2. Context: "Pre-existing conditions waiting period not stated"
       Acceptable: "Typical 24-48 month waiting period (inferred)"
    
    3. Context: "No mention of maternity coverage"
       Unacceptable: "Maternity covered at 100%"
       Acceptable: "Not specified in documents" 
    
    **Query**: {query}
    """
}

    def generate(self, query: str, retrieved_cases: List[RetrievalResult], insight_type: str = "analysis") -> GeneratedInsight:
        if not self.llm_initialized:
            return GeneratedInsight(
                content="LLM not initialized" if insight_type != "recommendation" else {"error": "LLM not initialized"},
                confidence=0.0,
                supporting_case_ids=[],
                insight_type=insight_type
            )

        # Optimize context - use top 3 cases only
        case_summaries = self._prepare_case_summaries(retrieved_cases[:3])
        
        prompt = self.prompt_templates[insight_type].format(
            query=query,
            case_summaries=case_summaries
        )
        
        # Track token usage
        self.token_counts.append(len(prompt.split()))
        logger.info(f"Prompt tokens: {len(prompt.split())}")

        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system", 
                        "content": "You output valid JSON for recommendation type" if insight_type == "recommendation" 
                                   else "You provide concise legal analysis"
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Lower for more deterministic output
                max_tokens=500,
                response_format={"type": "json_object"} if insight_type == "recommendation" else None
            )
            
            llm_output = response.choices[0].message.content
            logger.info(f"LLM response: {llm_output[:100]}...")
            
            # Parse JSON for recommendation type
            if insight_type == "recommendation":
                try:
                    llm_output = json.loads(llm_output)
                except json.JSONDecodeError:
                    logger.error("JSON parse failed, attempting repair")
                    llm_output = self._repair_json(llm_output)
        
        except Exception as e:
            logger.error(f"OpenAI error: {str(e)}")
            llm_output = {"error": "Generation failed"} if insight_type == "recommendation" else "Generation failed"

        verification = self._verify_insight(
            str(llm_output) if insight_type != "recommendation" else json.dumps(llm_output),
            retrieved_cases
        )

        return GeneratedInsight(
            content=llm_output,
            confidence=verification['confidence'],
            supporting_case_ids=verification['supporting_ids'],
            insight_type=insight_type
        )

    def _prepare_case_summaries(self, cases: List[RetrievalResult]) -> str:
        """Optimized to reduce token count"""
        return "\n".join(
            f"[{res.case.id}] Score: {res.score:.2f} | "
            f"{res.case.content[:150].strip()}{'...' if len(res.case.content) > 150 else ''}"
            for res in sorted(cases, key=lambda x: -x.score)
        )

    def _verify_insight(self, text: str, cases: List[RetrievalResult]) -> Dict:
        """Simplified verification"""
        case_ids = [res.case.id for res in cases]
        supporting_ids = [cid for cid in case_ids if f"[{cid}]" in text]
        confidence = min(0.95, len(supporting_ids) / len(cases) * 1.2) if cases else 0.0
        return {'confidence': round(confidence, 2), 'supporting_ids': supporting_ids}

    def _repair_json(self, text: str) -> Dict:
        """Basic JSON repair for common issues"""
        try:
            # Extract first JSON object
            start = text.find('{')
            end = text.rfind('}') + 1
            return json.loads(text[start:end])
        except:
            return {"error": "Invalid JSON response"}

    def get_avg_tokens(self) -> float:
        """Monitor performance"""
        return sum(self.token_counts) / len(self.token_counts) if self.token_counts else 0
