"""RAGAS evaluation. Runs every test question through the REAL pipeline (retrieval, guardrails,
prompt, LLM), then scores it with RAGAS. Exits non-zero if a score is under its threshold,
so it can gate CI/CD.

    pip install -r requirements-eval.txt
    python scripts/evaluate.py --testset eval/testset.jsonl

Test set: JSONL, one object per line:  {"question": "...", "reference": "ground-truth answer"}
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mmrag.config import get_settings  # noqa: E402
from mmrag.container import Container  # noqa: E402
from mmrag.logging_setup import setup_logging  # noqa: E402

# RAGAS column names differ slightly between versions, so accept aliases.
METRIC_COLUMNS = {
    "faithfulness": ["faithfulness"],
    "answer_relevancy": ["answer_relevancy", "response_relevancy"],
    "context_precision": ["llm_context_precision_with_reference", "context_precision"],
    "context_recall": ["context_recall", "llm_context_recall"],
}


def load_testset(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    for i, r in enumerate(rows, 1):
        if not r.get("question") or not r.get("reference"):
            raise SystemExit(f"{path}:{i} needs both 'question' and 'reference'")
    return rows


async def run_pipeline(c: Container, rows: list[dict]) -> list[dict]:
    samples = []
    for r in rows:
        a = await c.rag.answer_once(r["question"], session_id="eval")   # rate limiter bypassed
        samples.append({"user_input": r["question"], "response": a.answer,
                        "retrieved_contexts": a.contexts or ["(nothing retrieved)"],
                        "reference": r["reference"], "blocked": a.blocked})
        print(f"  answered: {r['question'][:70]}")
    return samples


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--testset", type=Path, default=Path("eval/testset.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("eval/results.csv"))
    ap.add_argument("--judge-model", default="gpt-4.1-mini", help="LLM used by RAGAS as the judge")
    ap.add_argument("--embedding-model", default="text-embedding-3-small")
    ap.add_argument("--min-faithfulness", type=float, default=0.80)
    ap.add_argument("--min-answer-relevancy", type=float, default=0.75)
    ap.add_argument("--min-context-precision", type=float, default=0.60)
    ap.add_argument("--min-context-recall", type=float, default=0.60)
    args = ap.parse_args()

    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas import EvaluationDataset, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall,
                               ResponseRelevancy)

    cfg = get_settings()
    setup_logging("WARNING")
    if cfg.openai_api_key is None:
        raise SystemExit("RAGAS evaluation requires OPENAI_API_KEY for its judge and embeddings.")
    key = cfg.openai_api_key.get_secret_value()
    c = Container(cfg)
    c.prepare()
    if c.db.stats()["documents_ready"] == 0:
        raise SystemExit("Nothing is indexed. Put PDFs in the inbox and run scripts/ingest.py first.")

    rows = load_testset(args.testset)
    print(f"Running {len(rows)} questions through the pipeline...")
    samples = asyncio.run(run_pipeline(c, rows))
    blocked = [s["user_input"] for s in samples if s["blocked"]]
    if blocked:
        print(f"WARNING: {len(blocked)} question(s) were blocked by guardrails: {blocked}")

    judge = LangchainLLMWrapper(ChatOpenAI(model=args.judge_model, temperature=0, api_key=key))
    embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=args.embedding_model, api_key=key))
    dataset = EvaluationDataset.from_list([{k: v for k, v in s.items() if k != "blocked"} for s in samples])

    print("Scoring with RAGAS...")
    result = evaluate(dataset=dataset,
                      metrics=[Faithfulness(), ResponseRelevancy(), LLMContextPrecisionWithReference(),
                               LLMContextRecall()],
                      llm=judge, embeddings=embeddings)
    df = result.to_pandas()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    thresholds = {"faithfulness": args.min_faithfulness, "answer_relevancy": args.min_answer_relevancy,
                  "context_precision": args.min_context_precision, "context_recall": args.min_context_recall}
    failed = False
    print(f"\n{'metric':<20}{'mean':>8}{'min':>8}{'threshold':>11}")
    for name, aliases in METRIC_COLUMNS.items():
        col = next((a for a in aliases if a in df.columns), None)
        if col is None:
            print(f"{name:<20}  (column not produced by this RAGAS version)")
            continue
        mean, low = df[col].mean(), df[col].min()
        ok = mean >= thresholds[name]
        failed |= not ok
        print(f"{name:<20}{mean:>8.3f}{low:>8.3f}{thresholds[name]:>11.2f}  {'PASS' if ok else 'FAIL'}")
    print(f"\nPer-question results: {args.out}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
