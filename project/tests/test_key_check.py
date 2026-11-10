"""Settings → 测试连接: a key is confirmed by asking the model service for its chat models."""

import httpx
import pytest

from core.user_settings import KeyCheckError, check_api_key


def _transport(status=200, models=("deepseek-ai/DeepSeek-V4-Flash", "Qwen/Qwen3-8B"), seen=None):
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        return httpx.Response(status, json={"data": [{"id": m} for m in models]})

    return httpx.MockTransport(handler)


def test_working_key_lists_models_and_reports_the_chosen_one():
    seen = []
    result = check_api_key("sk-good", "https://api.example.cn/v1", "Qwen/Qwen3-8B", transport=_transport(seen=seen))
    assert result["ok"] is True and result["model_available"] is True
    assert str(seen[0].url).startswith("https://api.example.cn/v1/models")
    assert seen[0].headers["Authorization"] == "Bearer sk-good"

    other = check_api_key("sk-good", "https://api.example.cn/v1/", "Qwen/Qwen3-235B", transport=_transport())
    assert other["model_available"] is False


@pytest.mark.parametrize(("status", "message"), [(401, "无效"), (403, "无效"), (500, "500")])
def test_rejected_key_says_why(status, message):
    with pytest.raises(KeyCheckError, match=message):
        check_api_key("sk-bad", "https://api.example.cn/v1", transport=_transport(status=status))


def test_unreachable_service_is_a_network_problem():
    def handler(request):
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(KeyCheckError, match="连不上"):
        check_api_key("sk-x", "https://api.example.cn/v1", transport=httpx.MockTransport(handler))
