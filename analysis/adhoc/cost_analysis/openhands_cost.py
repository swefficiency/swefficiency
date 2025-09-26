import json

jsonl_file = "predictions/openhands/claude37sonnet_raw.jsonl"


from decimal import Decimal
from typing import Dict, Optional

MTOK = Decimal(1_000_000)

# $ per MTok from your table
PRICING: Dict[str, Dict[str, Decimal]] = {
    "Claude Opus 4.1": {
        "base_in": Decimal("15"),
        "write_5m": Decimal("18.75"),
        "write_1h": Decimal("30"),
        "hit": Decimal("1.50"),
        "out": Decimal("75"),
    },
    "Claude Opus 4": {
        "base_in": Decimal("15"),
        "write_5m": Decimal("18.75"),
        "write_1h": Decimal("30"),
        "hit": Decimal("1.50"),
        "out": Decimal("75"),
    },
    "Claude Sonnet 4": {
        "base_in": Decimal("3"),
        "write_5m": Decimal("3.75"),
        "write_1h": Decimal("6"),
        "hit": Decimal("0.30"),
        "out": Decimal("15"),
    },
    "Claude Sonnet 3.7": {
        "base_in": Decimal("3"),
        "write_5m": Decimal("3.75"),
        "write_1h": Decimal("6"),
        "hit": Decimal("0.30"),
        "out": Decimal("15"),
    },
    "Claude Sonnet 3.5 (deprecated)": {
        "base_in": Decimal("3"),
        "write_5m": Decimal("3.75"),
        "write_1h": Decimal("6"),
        "hit": Decimal("0.30"),
        "out": Decimal("15"),
    },
    "Claude Haiku 3.5": {
        "base_in": Decimal("0.80"),
        "write_5m": Decimal("1"),
        "write_1h": Decimal("1.6"),
        "hit": Decimal("0.08"),
        "out": Decimal("4"),
    },
    "Claude Opus 3 (deprecated)": {
        "base_in": Decimal("15"),
        "write_5m": Decimal("18.75"),
        "write_1h": Decimal("30"),
        "hit": Decimal("1.50"),
        "out": Decimal("75"),
    },
    "Claude Haiku 3": {
        "base_in": Decimal("0.25"),
        "write_5m": Decimal("0.30"),
        "write_1h": Decimal("0.50"),
        "hit": Decimal("0.03"),
        "out": Decimal("1.25"),
    },
}

ALIASES = {
    # add whatever aliases you use; keys are lowercase
    "opus 4.1": "Claude Opus 4.1",
    "opus 4": "Claude Opus 4",
    "sonnet 4": "Claude Sonnet 4",
    "sonnet 3.7": "Claude Sonnet 3.7",
    "sonnet 3.5": "Claude Sonnet 3.5 (deprecated)",
    "haiku 3.5": "Claude Haiku 3.5",
    "opus 3": "Claude Opus 3 (deprecated)",
    "haiku 3": "Claude Haiku 3",
}


def _resolve_model(model: str) -> str:
    key = model.strip()
    if key in PRICING:
        return key
    alias = ALIASES.get(key.lower())
    if alias and alias in PRICING:
        return alias
    raise KeyError(f"Unknown model: {model}")


def price_request(
    *,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    cache_read_tokens: int = 0,  # cache hits + refreshes on this request
    cache_write_tokens_5m: int = 0,  # tokens written with 5m TTL on this request
    cache_write_tokens_1h: int = 0,  # tokens written with 1h TTL on this request
) -> Dict[str, Dict[str, Decimal]]:
    """
    Compute dollar cost for a single request using your pricing table.

    Billing decomposition:
      base_input = prompt_tokens - cache_read_tokens - cache_write_tokens_5m - cache_write_tokens_1h
      cost = base_input * base_in_rate
           + cache_write_tokens_5m * write_5m_rate
           + cache_write_tokens_1h * write_1h_rate
           + cache_read_tokens * hit_rate
           + completion_tokens * out_rate
      (all rates are $ per 1,000,000 tokens)

    Returns a dict with a breakdown and total (as Decimals).
    """
    m = _resolve_model(model)
    r = PRICING[m]

    # Validate and derive uncached base input
    base_input = (
        prompt_tokens
        - cache_read_tokens
        - cache_write_tokens_5m
        - cache_write_tokens_1h
    )
    if base_input < 0:
        raise ValueError(
            "Inconsistent token counts: base_input computed negative. "
            "Check prompt_tokens vs cache_read/write tokens."
        )

    base_cost = Decimal(base_input) * r["base_in"] / MTOK
    write5_cost = Decimal(cache_write_tokens_5m) * r["write_5m"] / MTOK
    write1_cost = Decimal(cache_write_tokens_1h) * r["write_1h"] / MTOK
    hit_cost = Decimal(cache_read_tokens) * r["hit"] / MTOK
    out_cost = Decimal(completion_tokens) * r["out"] / MTOK

    total = base_cost + write5_cost + write1_cost + hit_cost + out_cost

    return {
        "model": {"name": m},
        "tokens": {
            "base_input": Decimal(base_input),
            "cache_write_5m": Decimal(cache_write_tokens_5m),
            "cache_write_1h": Decimal(cache_write_tokens_1h),
            "cache_hits": Decimal(cache_read_tokens),
            "output": Decimal(completion_tokens),
            "prompt_total": Decimal(prompt_tokens),
        },
        "rates_per_MTok": {
            "base_in": r["base_in"],
            "write_5m": r["write_5m"],
            "write_1h": r["write_1h"],
            "hit": r["hit"],
            "out": r["out"],
        },
        "costs": {
            "base_input": base_cost,
            "cache_writes_5m": write5_cost,
            "cache_writes_1h": write1_cost,
            "cache_hits": hit_cost,
            "output": out_cost,
            "total": total,
        },
    }


# --- Convenience wrapper for your earlier usage object structure ---
def price_from_usage(
    usage: Dict[str, int], *, model: str, cache_write_ttl: Optional[str] = None
) -> Dict[str, Dict[str, Decimal]]:
    """
    usage expects keys:
      prompt_tokens, completion_tokens, cache_read_tokens, (optional) cache_write_tokens

    If cache_write_tokens is present and cache_write_ttl is set to '5m' or '1h',
    it will route those tokens to the corresponding bucket. If you used both TTLs,
    call price_request directly and pass each bucket explicitly.
    """
    prompt = int(usage.get("prompt_tokens", 0))
    output = int(usage.get("completion_tokens", 0))
    hits = int(usage.get("cache_read_tokens", 0))
    write = int(usage.get("cache_write_tokens", 0))

    w5 = write if cache_write_ttl == "5m" else 0
    w1 = write if cache_write_ttl == "1h" else 0

    return price_request(
        model=model,
        prompt_tokens=prompt,
        completion_tokens=output,
        cache_read_tokens=hits,
        cache_write_tokens_5m=w5,
        cache_write_tokens_1h=w1,
    )


total_cost = 0.0

line_counter = 0

per_token_breakdown = None

for line in open(jsonl_file):
    obj = json.loads(line)
    metrics = obj["metrics"]
    line_counter += 1

    if metrics is not None:
        total_cost += metrics["accumulated_cost"]

        # print(metrics["accumulated_token_usage"])
        if per_token_breakdown is None:
            per_token_breakdown = metrics["accumulated_token_usage"]
        else:
            for k, v in metrics["accumulated_token_usage"].items():
                if isinstance(v, (int, float)):
                    per_token_breakdown[k] += v

print(f"Processed {line_counter} lines.")
print(f"Total cost: ${total_cost:.2f}")
print("Per-token breakdown:", per_token_breakdown)
print(price_from_usage(per_token_breakdown, model="Claude Sonnet 3.7"))


print(price_from_usage(per_token_breakdown, model="Claude Opus 4.1"))
print(price_from_usage(per_token_breakdown, model="Claude Opus 4"))
