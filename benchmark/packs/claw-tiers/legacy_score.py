from pathlib import Path

def score_task(
    task: dict,
    final_text: str,
    tool_calls: list[dict],
    task_dir: Path,
) -> dict:
    """Apply rule-based scoring from task.yaml. Returns {score, details}.

    Supports check types: tool_called, keywords_present, categories_present, min_length.
    LLM judge tasks are skipped (return score=0, reason=llm_judge).
    """
    components = task.get("scoring_components", [])

    # Check for LLM judge tasks
    for comp in components:
        if comp.get("check", {}).get("type") == "llm_judge":
            return {"score": 0.0, "details": "skipped: llm_judge task not supported"}

    if not components:
        return {"score": 0.0, "details": "no scoring components"}

    total_weight = 0.0
    weighted_score = 0.0
    details_parts = []

    tool_names_called = {tc["tool"] for tc in tool_calls}

    for comp in components:
        name = comp.get("name", "?")
        weight = comp.get("weight", 0.0)
        check = comp.get("check", {})
        check_type = check.get("type", "")

        passed = False
        if check_type == "tool_called":
            target_tool = check.get("tool_name", "")
            min_calls = check.get("min_calls", 1)
            calls = sum(1 for tc in tool_calls if tc["tool"] == target_tool)
            passed = calls >= min_calls
            details_parts.append(
                f"{name}: {'PASS' if passed else 'FAIL'} ({target_tool} called {calls}/{min_calls})"
            )

        elif check_type == "keywords_present":
            keywords = [str(kw) for kw in check.get("keywords", [])]
            text_lower = str(final_text or "").lower()
            found = [kw for kw in keywords if kw.lower() in text_lower]
            passed = len(found) > 0
            details_parts.append(
                f"{name}: {'PASS' if passed else 'FAIL'} (keywords found: {found})"
            )

        elif check_type == "categories_present":
            categories = [str(cat) for cat in check.get("categories", [])]
            text_lower = str(final_text or "").lower()
            category_synonyms = {
                "fyi": ["fyi", "notifications", "notification", "info", "informational"],
                "notifications": ["notifications", "notification", "fyi", "info"],
                "action required": ["action required", "action items", "action item", "action"],
                "waiting for reply": ["waiting for reply", "waiting", "pending"],
            }
            found = []
            for cat in categories:
                cat_lower = cat.lower()
                syns = category_synonyms.get(cat_lower, [cat_lower])
                if any(syn in text_lower for syn in syns):
                    found.append(cat)
            passed = len(found) >= len(categories) * 0.5  # at least half
            details_parts.append(
                f"{name}: {'PASS' if passed else 'FAIL'} (categories: {found}/{categories})"
            )

        elif check_type == "min_length":
            field = check.get("field", "final_text")
            min_len = check.get("min_length", 0)
            text = str(final_text or "") if field == "final_text" else ""
            passed = len(text) >= min_len
            details_parts.append(
                f"{name}: {'PASS' if passed else 'FAIL'} (len={len(text)}/{min_len})"
            )

        else:
            details_parts.append(f"{name}: SKIP (unknown check type: {check_type})")

        if passed:
            weighted_score += weight
        total_weight += weight

    score = weighted_score / total_weight if total_weight > 0 else 0.0
    return {
        "score": round(score, 4),
        "details": " | ".join(details_parts),
        "tool_calls_count": len(tool_calls),
        "tools_used": sorted(tool_names_called),
        "final_text_length": len(final_text),
    }
