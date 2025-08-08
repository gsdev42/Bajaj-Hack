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
            'analysis': """*Task*: Analyze legal cases to answer: "{query}"
            
            *Cases* (sorted by relevance):
            {case_summaries}
            
            *Rules*:
            1. Use ONLY case content - no external knowledge
            2. Reference case IDs like [C001] where applicable
            3. Maximum 75 words
            
            *Output*: Concise analysis""",
            
            'recommendation': """*Role*: Insurance/legal expert assistant
            *Task*: Respond to query using ONLY the context below
            
            *Context*:
            {case_summaries}
            
            *Query*: {query}
            
            *Output Format* (JSON ONLY):
            {{
                "response_type": "decision|info|procedure|contact",
                "decision": "approved/rejected/N/A",
                "amount": "₹.../Covered/N/A",
                "answer": "1-sentence summary",
                "detailed_explanation": ["bullet 1", "bullet 2"],
                "referenced_clauses": ["Clause X.X"],
                "next_steps": ["action 1"],
                "baqis_score": 0-100
            }}
            
            *Rules*:
            1. Use ONLY context - if not found, use "N/A"
            2. baqis_score = confidence (0-100)
            3. MAX 5 bullet points
            4. Strict JSON format - NO additional text"""
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