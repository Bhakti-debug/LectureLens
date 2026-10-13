"""
LectureLens - Research Evaluation Framework
Evaluates Retrieval, Relevance Guard, and System Performance against a labelled benchmark dataset.
Produces reproducible JSON and CSV evaluation reports.
"""

import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure src is in sys.path
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from rag.pipeline import RAGPipeline

from services.lecture_service import get_lecture_service
from utils.config import get_config
from utils.logging_config import get_logger

logger = get_logger("evaluator")

# Ground-truth benchmark dataset for the sample lecture on Node.js
BENCHMARK_DATASET = [
    # 1. Direct Lecture Questions (in_scope)
    {
        "id": "q01",
        "question": "What is the name of Google Chrome's JavaScript engine?",
        "expected_category": "in_scope",
        "target_section": 1,
        "keywords": ["V8", "Google Chrome", "engine"],
    },
    {
        "id": "q02",
        "question": "Who created Node.js and what did he combine V8 with?",
        "expected_category": "in_scope",
        "target_section": 2,
        "keywords": ["Ryan Dahl", "C++", "V8"],
    },
    {
        "id": "q03",
        "question": "What happens when you run raw JavaScript code directly in a standard OS terminal without node?",
        "expected_category": "in_scope",
        "target_section": 3,
        "keywords": ["syntax error", "terminal", "interpreter", "REPL"],
    },
    {
        "id": "q04",
        "question": "What is the official definition of Node.js on nodejs.org?",
        "expected_category": "in_scope",
        "target_section": 4,
        "keywords": ["open-source", "cross-platform", "runtime environment"],
    },
    # 2. Paraphrased Questions (in_scope)
    {
        "id": "q05",
        "question": "How does bundling the V8 engine with C++ grant JavaScript access to the host operating system?",
        "expected_category": "in_scope",
        "target_section": 2,
        "keywords": ["native", "operating system", "hardware", "filesystem"],
    },
    {
        "id": "q06",
        "question": "Which JS engines are utilized by Mozilla Firefox and Apple Safari?",
        "expected_category": "in_scope",
        "target_section": 1,
        "keywords": ["SpiderMonkey", "JavaScriptCore"],
    },
    # 3. Conceptual Follow-ups (in_scope)
    {
        "id": "q07",
        "question": "Why is Node.js classified as a runtime environment rather than a programming language?",
        "expected_category": "in_scope",
        "target_section": 3,
        "keywords": ["runtime", "environment", "language"],
    },
    # 4. Insufficient Evidence (topic-related but not covered in video)
    {
        "id": "q08",
        "question": "What are the specific internal phases of the libuv event loop in Node.js?",
        "expected_category": "insufficient_evidence",
        "target_section": None,
        "keywords": ["libuv", "event loop", "phases", "timers"],
    },
    {
        "id": "q09",
        "question": "How do Worker Threads in Node.js communicate via MessagePort and SharedArrayBuffer?",
        "expected_category": "insufficient_evidence",
        "target_section": None,
        "keywords": ["Worker Threads", "SharedArrayBuffer", "MessagePort"],
    },
    # 5. Out of Scope / Unrelated Questions
    {
        "id": "q10",
        "question": "What is the capital city of Japan?",
        "expected_category": "out_of_scope",
        "target_section": None,
        "keywords": ["Tokyo", "Japan"],
    },
    {
        "id": "q11",
        "question": "What was the final score in the recent cricket world cup match?",
        "expected_category": "out_of_scope",
        "target_section": None,
        "keywords": ["cricket", "score"],
    },
    {
        "id": "q12",
        "question": "How do you cook traditional Italian carbonara pasta?",
        "expected_category": "out_of_scope",
        "target_section": None,
        "keywords": ["carbonara", "pasta", "recipe"],
    },
    # 6. Keyword-Sharing Trick Questions
    {
        "id": "q13",
        "question": "How does the Apple V8 internal combustion car engine compare to a Ferrari engine?",
        "expected_category": "out_of_scope",
        "target_section": None,
        "keywords": ["combustion", "Ferrari"],
    },
]


class LectureLensEvaluator:
    """Automated evaluation framework for retrieval and relevance."""

    def __init__(self, pipeline: RAGPipeline):
        self.pipeline = pipeline
        self.config = get_config()
        self.eval_dir = self.config.evaluations_dir
        self.eval_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_retrieval(self, dataset: List[Dict[str, Any]], k: int = 4) -> Dict[str, Any]:
        """
        Evaluate semantic retrieval accuracy: HitRate@k, MRR, Recall@k.
        Applies only to in-scope questions with a designated target_section.
        """
        target_questions = [q for q in dataset if q.get("target_section") is not None]
        if not target_questions or not self.pipeline.retriever:
            return {"hit_rate_at_k": 0.0, "mrr": 0.0, "total_queries": 0}

        hits = 0
        reciprocal_ranks = []
        latencies = []
        query_records = []

        for item in target_questions:
            q_text = item["question"]
            target_sec = item["target_section"]

            start_t = time.perf_counter()
            results = self.pipeline.retriever.retrieve(q_text, top_k=k)
            dur_ms = (time.perf_counter() - start_t) * 1000.0
            latencies.append(dur_ms)

            retrieved_sections = [r.chunk.section_number for r in results]
            is_hit = target_sec in retrieved_sections
            if is_hit:
                hits += 1
                rank = retrieved_sections.index(target_sec) + 1
                reciprocal_ranks.append(1.0 / rank)
            else:
                reciprocal_ranks.append(0.0)

            query_records.append(
                {
                    "id": item["id"],
                    "question": q_text,
                    "target_section": target_sec,
                    "retrieved_sections": str(retrieved_sections),
                    "hit": is_hit,
                    "mrr": round(reciprocal_ranks[-1], 3),
                    "latency_ms": round(dur_ms, 2),
                }
            )

        hit_rate = round(hits / len(target_questions), 4)
        mrr = round(sum(reciprocal_ranks) / len(reciprocal_ranks), 4)
        avg_lat = round(sum(latencies) / len(latencies), 2)

        # Write CSV report
        csv_file = self.eval_dir / "retrieval_metrics.csv"
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["id", "question", "target_section", "retrieved_sections", "hit", "mrr", "latency_ms"],
            )
            writer.writeheader()
            writer.writerows(query_records)

        return {
            "total_queries": len(target_questions),
            "hit_rate_at_k": hit_rate,
            "mrr": mrr,
            "avg_latency_ms": avg_lat,
            "k": k,
        }

    def evaluate_relevance(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate relevance classification accuracy and confusion matrix:
        - True Positives (in_scope correctly accepted)
        - False Positives (out_of_scope falsely accepted)
        - True Negatives (out_of_scope correctly rejected)
        - False Negatives (in_scope falsely rejected)
        """
        guard = self.pipeline.relevance_guard
        if not self.pipeline.retriever:
            return {"accuracy": 0.0}

        tp = 0  # In-scope accepted
        fp = 0  # Out-of-scope accepted
        tn = 0  # Out-of-scope rejected
        fn = 0  # In-scope rejected

        details = []

        for item in dataset:
            q_text = item["question"]
            expected_cat = item["expected_category"]
            should_accept = expected_cat in ("in_scope", "insufficient_evidence")

            # Retrieve evidence for guard
            retrieved = self.pipeline.retriever.retrieve(q_text, top_k=4)

            eval_res = guard.evaluate_query(
                question=q_text,
                lecture_title=self.pipeline.lecture_title,
                lecture_overview=self.pipeline.lecture_overview,
                key_topics=self.pipeline.key_topics,
                retrieved_results=retrieved,
            )

            actual_accepted = eval_res.get("is_relevant", True)
            actual_cat = eval_res.get("category", "in_scope")

            if should_accept and actual_accepted:
                tp += 1
                classification = "TP"
            elif not should_accept and actual_accepted:
                fp += 1
                classification = "FP"
            elif not should_accept and not actual_accepted:
                tn += 1
                classification = "TN"
            else:
                fn += 1
                classification = "FN"

            details.append(
                {
                    "id": item["id"],
                    "question": q_text,
                    "expected": expected_cat,
                    "predicted_category": actual_cat,
                    "predicted_relevant": actual_accepted,
                    "result": classification,
                    "reason": eval_res.get("reason", ""),
                }
            )

        total = len(dataset)
        accuracy = round((tp + tn) / total, 4) if total > 0 else 0.0
        precision = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
        recall = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0
        f1 = (
            round(2 * (precision * recall) / (precision + recall), 4)
            if (precision + recall) > 0
            else 0.0
        )

        return {
            "total_samples": total,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": {
                "true_positives": tp,
                "false_positives": fp,
                "true_negatives": tn,
                "false_negatives": fn,
            },
            "false_rejection_rate": round(fn / (tp + fn), 4) if (tp + fn) > 0 else 0.0,
            "false_acceptance_rate": round(fp / (tn + fp), 4) if (tn + fp) > 0 else 0.0,
            "details": details,
        }

    def run_full_evaluation(self) -> Dict[str, Any]:
        """Execute complete research evaluation and save summary report."""
        logger.info("Running retrieval evaluation...")
        retrieval_res = self.evaluate_retrieval(BENCHMARK_DATASET)

        logger.info("Running relevance guard evaluation...")
        relevance_res = self.evaluate_relevance(BENCHMARK_DATASET)

        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "lecture_id": self.pipeline.lecture_id,
            "lecture_title": self.pipeline.lecture_title,
            "embedding_model": self.pipeline.embedding_service.model_name,
            "retrieval_metrics": retrieval_res,
            "relevance_metrics": {
                k: v for k, v in relevance_res.items() if k != "details"
            },
            "relevance_query_details": relevance_res.get("details", []),
        }

        report_file = self.eval_dir / "evaluation_report.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"Full evaluation report saved to: {report_file}")
        return report


def main():
    """CLI runner for benchmark evaluation."""
    service = get_lecture_service()
    lecture_data = service.load_lecture("sample_node_lecture")

    if not lecture_data.get("analysis"):
        print("Error: No sample lecture found. Please ensure data/analysis/video_analysis.json is present.")
        return

    analysis = lecture_data["analysis"]
    notes = lecture_data["notes"]
    title = notes.get("title") if notes else "Introduction to Node.js"

    pipeline = RAGPipeline(
        lecture_id="sample_node_lecture",
        lecture_title=title,
    )

    # Build or verify index
    indexed_count = pipeline.index_lecture(analysis_data=analysis, notes_data=notes)
    print(f"Index ready: {indexed_count} chunks indexed.")

    evaluator = LectureLensEvaluator(pipeline)
    print("Executing benchmark evaluation...")
    results = evaluator.run_full_evaluation()

    print("\n================== EVALUATION REPORT ==================")
    print(f"Retrieval HitRate@4: {results['retrieval_metrics'].get('hit_rate_at_k')}")
    print(f"Retrieval MRR:       {results['retrieval_metrics'].get('mrr')}")
    print(f"Retrieval Latency:   {results['retrieval_metrics'].get('avg_latency_ms')} ms")
    print("-------------------------------------------------------")
    print(f"Relevance Accuracy:  {results['relevance_metrics'].get('accuracy')}")
    print(f"Relevance Precision: {results['relevance_metrics'].get('precision')}")
    print(f"Relevance Recall:    {results['relevance_metrics'].get('recall')}")
    print(f"Relevance F1 Score:  {results['relevance_metrics'].get('f1_score')}")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
