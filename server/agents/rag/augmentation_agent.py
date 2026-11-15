from __future__ import annotations

from typing import Any, Dict, List, Optional

from server.utils.gemini_client import generate_text_with_config, has_gemini_api_key


class AugmentationAgent:
    def augment(
        self,
        patient_id: str,
        query: str,
        chunks: List[Dict[str, Any]],
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, str]:
        clean_q = (query or "").strip()
        expanded_query = clean_q or "Provide a concise overview of this case."

        if clean_q and history and has_gemini_api_key():
            try:
                hist_str = "\n".join(
                    f"{m.get('role', 'user')}: {m.get('content', '')}"
                    for m in history[-6:]
                )
                prompt = (
                    "Rewrite the user's latest question into a standalone search query "
                    "incorporating context from the conversation history. Return ONLY the query.\n\n"
                    f"History:\n{hist_str}\n\nLatest Question: {clean_q}"
                )
                out = generate_text_with_config(
                    "rag_augmentation",
                    prompt,
                    "You expand clinical case questions into standalone retrieval queries.",
                )
                if out:
                    expanded_query = out
            except Exception:
                pass

        context_block = (
            "\n\n---\n\n".join(f"[{c['chunk_id']}]\n{c['text']}" for c in chunks)
            if chunks
            else "No indexed report chunks are currently available for this patient."
        )

        system_prompt = (
            f'You are the MedCore Case Assistant for patient "{patient_id}".\n'
            f'Answer ONLY using the retrieved report chunks and tool outputs for patient "{patient_id}".\n'
            "Never reference, infer, or leak data from any other patient.\n"
            "Cite chunk IDs inline like [RPT-...#chunk-1] whenever quoting report facts.\n"
            "If the requested report does not exist or lacks the answer, state that clearly.\n\n"
            f"RETRIEVED CASE CONTEXT:\n{context_block}"
        )

        return {"expanded_query": expanded_query, "system_prompt": system_prompt}


def self_test() -> None:
    agent = AugmentationAgent()
    out = agent.augment(
        patient_id="PAT-1001",
        query="Why was it approved?",
        chunks=[{"chunk_id": "RPT-1#chunk-1", "text": "Determination: APPROVED"}],
        history=[{"role": "user", "content": "Tell me about Elena Morales MRI"}],
    )
    assert len(out["expanded_query"]) > 0
    assert "PAT-1001" in out["system_prompt"]
    assert "RPT-1#chunk-1" in out["system_prompt"]
    print("[PASS] AugmentationAgent self_test")


if __name__ == "__main__":
    self_test()
