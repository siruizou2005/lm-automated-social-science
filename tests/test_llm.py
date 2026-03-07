import json

from src.LLM.LLM import LanguageModel, llm_json_loader


def test_llm_json_loader_normalizes_smart_quotes():
    parsed = llm_json_loader('{”answer”:”1”}')
    assert parsed["answer"] == "1"


def test_replicate_family_does_not_require_openai_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ORGANIZATION_ID", raising=False)

    llm = LanguageModel(
        family="replicate",
        model="llama13b-v2-chat",
        temperature=0.1,
    )

    assert llm.family == "replicate"
    assert llm.client is None
    assert json.loads(llm.serialize())["args"]["family"] == "replicate"
