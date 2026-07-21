import json
import os
from fastapi import FastAPI, Request
from openai import OpenAI

app = FastAPI()

client = OpenAI(
    api_key=os.environ["AIPIPE_TOKEN"],
    base_url="https://aipipe.org/openai/v1",
)

SYSTEM_PROMPT = (
    "You extract structured invoice data from messy text. "
    "CRITICAL: for `vendor`, `sku`, and any other field copied directly "
    "from the text, reproduce the exact characters as they appear in the "
    "source — do not add, remove, or normalize punctuation (e.g. do not "
    "add or remove trailing periods like 'Co' vs 'Co.'), do not fix "
    "spelling, do not expand or add abbreviations. Copy verbatim.\n"
    "- currency: convert to ISO 4217 code (USD, EUR, GBP, INR, JPY)\n"
    "- total_amount: plain integer, no separators, no symbols, "
    "convert spelled-out numbers, K/M suffixes, and Indian-style "
    "grouping (1,24,800) to a normal integer\n"
    "- invoice_date: YYYY-MM-DD\n"
    "- due_in_days: integer, convert phrases like 'Net 30' or "
    "'two weeks' to a day count\n"
    "- is_paid: boolean inferred from wording\n"
    "- priority: one of low, normal, high, urgent\n"
    "- contact_email: lowercase\n"
    "- line_items: array in the order they appear, unit_price "
    "as integer\n"
    "- item_count: number of line items\n"
    "Return ONLY the JSON object, matching the schema exactly. "
    "No extra keys, no missing keys, no commentary."
)


def clean_trailing_punct(value: str) -> str:
    """Strip a single trailing period that's likely sentence punctuation
    rather than part of the actual name (e.g. 'Meridian Paper Co.' at the
    end of a sentence -> 'Meridian Paper Co')."""
    if isinstance(value, str):
        return value.rstrip(".").strip()
    return value


def postprocess(result: dict) -> dict:
    if "vendor" in result:
        result["vendor"] = clean_trailing_punct(result["vendor"])

    if "contact_email" in result and isinstance(result["contact_email"], str):
        result["contact_email"] = result["contact_email"].lower()

    if "line_items" in result and isinstance(result["line_items"], list):
        for item in result["line_items"]:
            if "sku" in item:
                item["sku"] = clean_trailing_punct(item["sku"])
        result["item_count"] = len(result["line_items"])

    return result


@app.post("/extract")
async def extract(request: Request):
    body = await request.json()
    text = body["text"]
    schema = body["schema"]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "invoice",
                "schema": schema,
                "strict": True,
            },
        },
    )

    result = json.loads(response.choices[0].message.content)
    result = postprocess(result)
    return result