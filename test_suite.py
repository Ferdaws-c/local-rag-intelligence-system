"""
test_suite.py — Functional Test Suite for the Local RAG Intelligence System
=======================================================================
Week 5: System Testing & Evaluation

Tests three categories of queries:
  1. IN-CONTEXT    : Questions the assistant SHOULD answer correctly.
  2. OUT-OF-CONTEXT: Questions it should NOT answer (fallback expected).
  3. EDGE CASES    : Empty input, very short input, etc.

Usage:
    python test_suite.py

Output:
    Prints a detailed report to the console and saves results to
    test_results.txt for documentation purposes.
"""

import time
import sys
from pathlib import Path
from datetime import datetime

from sdk_utils import init_sdk, load_model
from rag_core import answer_query

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
BASE_DIR        = Path(__file__).parent
EMBEDDING_MODEL = "qwen3-embedding-0.6b"
CHAT_MODEL      = "phi-3.5-mini"   # Balanced model for testing
RESULTS_FILE    = BASE_DIR / "test_results.txt"

# ------------------------------------------------------------------
# Test Cases
# ------------------------------------------------------------------
# Format: (query, expected_behaviour, category)
# expected_behaviour is a keyword/phrase to look for in the answer.
# For OUT-OF-CONTEXT, we expect the "don't have that information" fallback.

TEST_CASES = [
    # --- IN-CONTEXT: Should answer from documents ---
    (
        "Who is Ferdaws Qaem?",
        "Ferdaws",
        "IN-CONTEXT",
    ),
    (
        "What university does Ferdaws attend?",
        "Kültür",
        "IN-CONTEXT",
    ),
    (
        "What is Ferdaws's student ID?",
        "2300001530",
        "IN-CONTEXT",
    ),
    (
        "What languages does Ferdaws speak?",
        "English",
        "IN-CONTEXT",
    ),
    (
        "What was Ferdaws's grade for Programming I?",
        "B",
        "IN-CONTEXT",
    ),
    (
        "Has Ferdaws completed an Erasmus program?",
        "Erasmus",
        "IN-CONTEXT",
    ),

    # --- OUT-OF-CONTEXT: Should return the fallback message ---
    (
        "What is the current price of Bitcoin?",
        "don't have that information",
        "OUT-OF-CONTEXT",
    ),
    (
        "Who is the CEO of Microsoft?",
        "don't have that information",
        "OUT-OF-CONTEXT",
    ),
    (
        "Can you write me a poem?",
        "don't have that information",
        "OUT-OF-CONTEXT",
    ),

    # --- EDGE CASES ---
    (
        "hi",
        "don't have that information",
        "EDGE-CASE",
    ),
    (
        "?",
        "don't have that information",
        "EDGE-CASE",
    ),
    (
        "Tell me everything about everything",
        "Microsoft",
        "EDGE-CASE",
    ),
]


# ------------------------------------------------------------------
# Test Runner
# ------------------------------------------------------------------
def run_tests(embedding_client, chat_client) -> list[dict]:
    """
    Runs all test cases and returns a list of result dicts.
    """
    results = []
    total   = len(TEST_CASES)

    print(f"\n{'='*60}")
    print(f"  Running {total} test cases...")
    print(f"{'='*60}\n")

    for i, (query, expected_keyword, category) in enumerate(TEST_CASES, 1):
        print(f"[{i:02d}/{total}] {category}: \"{query[:55]}...\"" if len(query) > 55 else f"[{i:02d}/{total}] {category}: \"{query}\"")

        # Skip empty queries
        if not query.strip():
            results.append({
                "query":    query,
                "category": category,
                "answer":   "(empty input — skipped)",
                "passed":   True,
                "elapsed":  0.0,
            })
            print(f"       SKIP — empty input\n")
            continue

        start   = time.time()
        result  = answer_query(
            question=query,
            embedding_client=embedding_client,
            chat_client=chat_client,
            top_k=3,
        )
        elapsed = time.time() - start
        answer  = result["answer"]

        # Pass/fail check: does the answer contain the expected keyword?
        passed = expected_keyword.lower() in answer.lower()
        status = "PASS ✅" if passed else "FAIL ❌"

        print(f"       {status}  ({elapsed:.1f}s)")
        print(f"       Expected to contain : \"{expected_keyword}\"")
        print(f"       Got                 : \"{answer[:100]}...\"" if len(answer) > 100 else f"       Got                 : \"{answer}\"")
        print()

        results.append({
            "query":    query,
            "category": category,
            "answer":   answer,
            "passed":   passed,
            "elapsed":  elapsed,
        })

    return results


# ------------------------------------------------------------------
# Report Generation
# ------------------------------------------------------------------
def print_summary(results: list[dict]) -> str:
    """Prints and returns a formatted summary report."""
    total   = len(results)
    passed  = sum(1 for r in results if r["passed"])
    failed  = total - passed
    avg_sec = sum(r["elapsed"] for r in results) / total if total else 0

    # Category breakdown
    categories = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"pass": 0, "fail": 0}
        if r["passed"]:
            categories[cat]["pass"] += 1
        else:
            categories[cat]["fail"] += 1

    lines = []
    lines.append("\n" + "=" * 60)
    lines.append("  TEST RESULTS SUMMARY")
    lines.append("=" * 60)
    lines.append(f"  Date/Time  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  Chat Model : {CHAT_MODEL}")
    lines.append(f"  Total Tests: {total}")
    lines.append(f"  Passed     : {passed}  |  Failed: {failed}")
    lines.append(f"  Pass Rate  : {passed/total*100:.0f}%")
    lines.append(f"  Avg Speed  : {avg_sec:.1f}s per query")
    lines.append("")
    lines.append("  Breakdown by category:")
    for cat, counts in categories.items():
        lines.append(f"    {cat:<20} Pass: {counts['pass']}  Fail: {counts['fail']}")

    if failed > 0:
        lines.append("")
        lines.append("  Failed queries:")
        for r in results:
            if not r["passed"]:
                lines.append(f"    ✗ [{r['category']}] {r['query']}")
                lines.append(f"      Got: {r['answer'][:80]}...")

    lines.append("=" * 60)

    report = "\n".join(lines)
    print(report)
    return report


# ------------------------------------------------------------------
# Entry Point
# ------------------------------------------------------------------
def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
    print("=" * 60)
    print("  Local RAG Intelligence System — Automated Test Suite (Week 5)")
    print("=" * 60)

    # Load models
    print(f"\nLoading models...")
    manager          = init_sdk("local_rag_assistant")
    embedding_model  = load_model(manager, EMBEDDING_MODEL, "embedding model")
    embedding_client = embedding_model.get_embedding_client()
    chat_model       = load_model(manager, CHAT_MODEL, "chat model")
    chat_client      = chat_model.get_chat_client()
    print("Models ready.\n")

    # Run tests
    results = run_tests(embedding_client, chat_client)

    # Print and save summary
    report = print_summary(results)

    RESULTS_FILE.write_text(report, encoding="utf-8")
    print(f"\n📄 Full report saved to: {RESULTS_FILE}")

    # Cleanup
    embedding_model.unload()
    chat_model.unload()


if __name__ == "__main__":
    main()
