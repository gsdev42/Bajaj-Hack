import numpy as np
from typing import List, Dict, Any  
from dataclasses import dataclass
import logging
from transformers import pipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CaseDocument:
    id: str
    content: str
    vector: np.ndarray
    metadata: Dict[str, Any]  

@dataclass
class RetrievalResult:
    case: CaseDocument
    score: float
    rank: int = 0

@dataclass
class GeneratedInsight:
    content: str
    confidence: float
    supporting_case_ids: List[str]
    insight_type: str

class CaseInsightGenerator:
    def __init__(self, model_name: str = "gpt2"):
        try:
            self.llm = pipeline("text-generation", model=model_name)
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {str(e)}")
            self.llm = None

        self.prompt_templates = {
            'analysis': """Analyze these {num_cases} legal cases for: {query}
            Cases: {case_summaries}
            Provide analysis with direct references to case IDs like [C001].""",
            'recommendation': """Based on cases: {case_summaries}
            Generate 3-5 actionable recommendations for: {query}
            Cite relevant cases like [C002] where applicable."""
        }

    def generate(self, query: str, retrieved_cases: List[RetrievalResult], insight_type: str = "analysis") -> GeneratedInsight:
        if not self.llm:
            return GeneratedInsight(
                content="LLM not initialized",
                confidence=0.0,
                supporting_case_ids=[],
                insight_type=insight_type
            )

        case_summaries = self._prepare_case_summaries(retrieved_cases)
        
        prompt = self.prompt_templates[insight_type].format(
            query=query,
            case_summaries=case_summaries,
            num_cases=len(retrieved_cases)
        )
        
        try:
            llm_output = self.llm(prompt, max_length=500, do_sample=True)[0]['generated_text']
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            llm_output = "Insight generation failed"

        verification = self._verify_insight(llm_output, retrieved_cases)
        
        return GeneratedInsight(
            content=llm_output,
            confidence=verification['confidence'],
            supporting_case_ids=verification['supporting_ids'],
            insight_type=insight_type
        )

    def _prepare_case_summaries(self, cases: List[RetrievalResult]) -> str:
        return "\n".join(
            f"[{res.score:.2f}] {res.case.id}: {res.case.content[:200]}..."
            for res in sorted(cases, key=lambda x: -x.score)
        )

    def _verify_insight(self, text: str, cases: List[RetrievalResult]) -> Dict:
        case_ids = [res.case.id for res in cases]
        supporting_ids = [cid for cid in case_ids if f"[{cid}]" in text]
        confidence = min(0.95, len(supporting_ids) / len(cases) * 1.5) if cases else 0.0
        
        return {
            'confidence': round(confidence, 2),
            'supporting_ids': supporting_ids
        }
if __name__ == "__main__":
    print("=== TESTING INSIGHT GENERATION (No LLM) ===")
    
    # Create mock generator without LLM
    generator = CaseInsightGenerator()
    generator.llm = None  
    
    mock_cases = [
        RetrievalResult(
            case=CaseDocument(
                id="C001",
                content="Sample case content about data privacy...",
                vector=np.array([0.1]*768),
                metadata={"source": "EU"}
            ),
            score=0.9,
            rank=1
        )
    ]
    
    insight = generator.generate(
        query="GDPR requirements",
        retrieved_cases=mock_cases,
        insight_type="recommendation"
    )
    
    print("\n=== TEST OUTPUT ===")
    print(f"Type: {insight.insight_type}")
    print(f"Confidence: {insight.confidence:.0%}")
    print(f"Supported by: {insight.supporting_case_ids}")
    print("\nContent Preview:")
    print(insight.content[:200] + "...")
