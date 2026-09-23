"""
reflexion_loop.py — Self-Correction & Reflexion Loop Engine.

Monitors NLI faithfulness scores. If score < 0.70, automatically reformulates queries,
re-retrieves expanded PubMed evidence, and regenerates clinical explanation until verified.
"""

from typing import Dict, List, Tuple
from src.retrieval import PubMedRetriever
from src.generation import generate_explanation
from src.faithfulness import evaluate_faithfulness


class ReflexionSelfCorrectionLoop:
    """Reflexion Loop for automated explanation self-correction."""

    def __init__(self, disease_name: str, min_faithfulness: float = 0.70, max_iterations: int = 2):
        self.disease_name = disease_name
        self.min_faithfulness = min_faithfulness
        self.max_iterations = max_iterations
        self.retriever = PubMedRetriever(disease_name)

    def execute_reflexion(
        self,
        predicted_class: str,
        confidence: float,
        initial_explanation: str,
        initial_evidence: List[Dict],
        initial_faithfulness: float,
    ) -> Tuple[str, List[Dict], Dict, bool]:
        """
        Executes self-correction loop if initial faithfulness < min_faithfulness.
        """
        current_explanation = initial_explanation
        current_evidence = initial_evidence
        current_score = initial_faithfulness
        self_corrected = False

        iteration = 0
        while current_score < self.min_faithfulness and iteration < self.max_iterations:
            iteration += 1
            self_corrected = True
            print(f"  [REFLEXION] Loop Iteration {iteration}: Initial score ({current_score:.4f}) < {self.min_faithfulness}. Re-retrieving expanded PubMed evidence...")

            # 1. Query Reformulation & Expanded Re-retrieval
            expanded_query = f"{self.disease_name.replace('_', ' ')} {predicted_class} diagnosis pathology clinical management risk"
            new_evidence = self.retriever.retrieve(expanded_query, top_k_final=5)
            if new_evidence:
                current_evidence = new_evidence

            # 2. Re-generate Explanation
            current_explanation = generate_explanation(self.disease_name, predicted_class, confidence, current_evidence)

            # 3. Re-evaluate Faithfulness
            faith_res = evaluate_faithfulness(current_explanation, current_evidence)
            current_score = faith_res['average_faithfulness']

        final_faith_res = evaluate_faithfulness(current_explanation, current_evidence)
        return current_explanation, current_evidence, final_faith_res, self_corrected


if __name__ == '__main__':
    print("Reflexion loop module ready.")
