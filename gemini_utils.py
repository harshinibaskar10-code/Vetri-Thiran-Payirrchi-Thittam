"""AI orchestration for PocketSmart.

* builds prompts for the Home, Party and Jewelry planners
* calls Gemini (text, or text + image for the jewelry planner)
* validates / repairs the JSON so totals always add up and never exceed the budget
* adds shopping-search links for Indian platforms
* falls back to a rule-based plan when Gemini is unavailable or returns nothing usable
"""
import json
import logging
import math
import re
import urllib.parse
from typing import Any, Callable, Dict, List, Optional, Tuple

import config
from models import HomeBudgetInput, JewelryBudgetInput, PartyBudgetInput

log = logging.getLogger("pocketsmart.ai")

# --------------------------------------------------------------------------- #
# Gemini client
# --------------------------------------------------------------------------- #
_client = None


def ai_enabled() -> bool:
    return bool(config.GEMINI_API_KEY) and not config.FORCE_MOCK


def _get_client():
    global _client
    if _client is None:
        from google import genai

        _client = genai.Client(api_key=config.GEMINI_API_KEY)
    return _client


def extract_json_from_response(text: str) -> Dict[str, Any]:
    """Parse JSON from a model reply, tolerating ```json fences and extra prose."""
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start : end + 1])
    raise ValueError("Model did not return valid JSON")


def call_gemini(prompt: str, image_bytes: Optional[bytes] = None, mime_type: Optional[str] = None) -> Dict[str, Any]:
    from google.genai import types

    parts: List[Any] = [prompt]
    if image_bytes:
        parts.append(types.Part.from_bytes(data=image_bytes, mime_type=mime_type or "image/jpeg"))
    response = _get_client().models.generate_content(
        model=config.GEMINI_MODEL,
        contents=parts,
        config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.4),
    )
    return extract_json_from_response(response.text)


def _generate(prompt: str, fallback: Callable[[], Dict[str, Any]],
              image_bytes: Optional[bytes] = None, mime: Optional[str] = None) -> Tuple[Dict[str, Any], str, str]:
    """Return (raw_result, source, reason). Source is 'gemini' or 'fallback'."""
    reason = ""
    if not ai_enabled():
        reason = "No GEMINI_API_KEY configured, so this is an offline sample plan."
    else:
        for attempt in (1, 2):
            try:
                raw = call_gemini(prompt, image_bytes, mime)
                if isinstance(raw, dict):
                    return raw, "gemini", ""
                reason = "Gemini returned an unexpected response format."
            except Exception as exc:  # network, quota, auth, bad JSON ...
                log.warning("Gemini attempt %s failed: %s", attempt, exc)
                reason = f"Gemini call failed: {exc}"
    return fallback(), "fallback", reason[:300]


# --------------------------------------------------------------------------- #
# Shopping links (search URLs – we do not scrape or call retailer APIs)
# --------------------------------------------------------------------------- #
PLATFORM_URLS = {
    "amazon": "https://www.amazon.in/s?k={q}",
    "flipkart": "https://www.flipkart.com/search?q={q}",
    "ikea": "https://www.ikea.com/in/en/search/?q={q}",
    "pepperfry": "https://www.pepperfry.com/site_product/search?q={q}",
    "bigbasket": "https://www.bigbasket.com/ps/?q={q}",
    "swiggy": "https://www.swiggy.com/search?query={q}",
    "zomato": "https://www.zomato.com/search?q={q}",
    "bookmyshow": "https://in.bookmyshow.com/search?q={q}",
    "meesho": "https://www.meesho.com/search?q={q}",
    "google": "https://www.google.com/search?q={q}",
    "booking": "https://www.booking.com/searchresults.html?ss={q}",
    "makemytrip": "https://www.makemytrip.com/hotels/hotel-listing/?searchText={q}",
    "oyorooms": "https://www.oyorooms.com/search/?location={q}",
    "bluestone": "https://www.bluestone.com/search.html?query={q}",
    "tanishq": "https://www.tanishq.co.in/search?q={q}",
    "caratlane": "https://www.caratlane.com/search?q={q}",
    "melorra": "https://www.melorra.com/search?q={q}",
}

HOME_PLATFORMS = ["amazon", "flipkart", "ikea", "pepperfry"]
JEWELRY_PLATFORMS = ["amazon", "flipkart", "bluestone", "tanishq", "caratlane", "melorra", "meesho"]
VENUE_PLATFORMS = ["google", "booking", "makemytrip", "oyorooms"]
PARTY_PLATFORMS = {
    "venue": VENUE_PLATFORMS,
    "catering": ["swiggy", "zomato"],
    "food": ["swiggy", "zomato", "bigbasket"],
    "decoration": ["amazon", "flipkart", "meesho"],
    "entertainment": ["bookmyshow", "amazon", "flipkart"],
    "gifts": ["amazon", "flipkart", "meesho"],
    "return_gifts": ["amazon", "flipkart", "meesho"],
}
DEFAULT_PARTY_PLATFORMS = ["amazon", "flipkart", "google"]
NO_LINK_CATEGORIES = {"contingency", "misc", "buffer"}


def build_links(platforms: List[str], terms: str) -> Dict[str, str]:
    qs = urllib.parse.quote_plus(terms.strip())
    return {p: PLATFORM_URLS[p].format(q=qs) for p in platforms if p in PLATFORM_URLS}


# --------------------------------------------------------------------------- #
# Number / structure helpers
# --------------------------------------------------------------------------- #
def _num(value: Any, default: float = 0.0) -> float:
    try:
        return float(str(value).replace(",", "").replace("₹", "").strip())
    except (TypeError, ValueError):
        return default


def _prepare_items(items: Any) -> List[Dict[str, Any]]:
    """Keep only dict items and coerce numeric fields."""
    out = []
    for it in items or []:
        if not isinstance(it, dict):
            continue
        item = dict(it)
        item["name"] = str(it.get("name") or it.get("item_type") or "Item")
        item["description"] = str(it.get("description") or "")
        item["estimated_price"] = max(_num(it.get("estimated_price")), 0.0)
        item["quantity"] = max(int(_num(it.get("quantity"), 1)), 1)
        item["search_terms"] = str(it.get("search_terms") or "")
        out.append(item)
    return out


def _fit_to_budget(items: List[Dict[str, Any]], budget: float) -> List[str]:
    """Scale unit prices down if the plan exceeds the budget."""
    total = sum(i["estimated_price"] * i["quantity"] for i in items)
    if total > budget and total > 0:
        factor = budget / total
        for i in items:
            i["estimated_price"] = math.floor(i["estimated_price"] * factor * 100) / 100
        return [f"Prices were scaled down by about {round((1 - factor) * 100)}% so the plan stays within your budget."]
    return []


def _str_list(value: Any) -> List[str]:
    return [str(v) for v in value if v] if isinstance(value, list) else []


def _finalize_breakdown(raw: Dict[str, Any], budget: float) -> Dict[str, Any]:
    """Recompute every total ourselves instead of trusting the model's arithmetic."""
    cats = [c for c in (raw.get("budget_breakdown") or []) if isinstance(c, dict)]
    all_items: List[Dict[str, Any]] = []
    for c in cats:
        c["items"] = _prepare_items(c.get("items"))
        c["category"] = str(c.get("category") or "misc")
        all_items += c["items"]
    notes = _fit_to_budget(all_items, budget)

    table, spent = [], 0.0
    for c in cats:
        cat_total = 0.0
        for i in c["items"]:
            i["line_total"] = round(i["estimated_price"] * i["quantity"], 2)
            cat_total += i["line_total"]
        c["allocation"] = round(cat_total, 2)
        spent += cat_total
        table.append({
            "category": c["category"],
            "items_count": len(c["items"]),
            "total_cost": round(cat_total, 2),
            "percentage_of_budget": round(cat_total / budget * 100, 1),
        })
    return {
        "total_budget": round(budget, 2),
        "budget_breakdown": cats,
        "calculation_table": table,
        "total_spent": round(spent, 2),
        "remaining_budget": round(budget - spent, 2),
        "additional_suggestions": _str_list(raw.get("additional_suggestions")) + notes,
    }


def _breakdown_has_items(raw: Dict[str, Any]) -> bool:
    cats = raw.get("budget_breakdown")
    return isinstance(cats, list) and any(isinstance(c, dict) and c.get("items") for c in cats)


def _add_links(result: Dict[str, Any], platforms_for: Callable[[str], List[str]]) -> None:
    for c in result["budget_breakdown"]:
        key = c["category"].lower().replace(" ", "_")
        for i in c["items"]:
            terms = i["search_terms"] or ("" if key in NO_LINK_CATEGORIES else i["name"])
            i["shopping_links"] = build_links(platforms_for(key), terms) if terms else {}


def _tag(result: Dict[str, Any], source: str, reason: str) -> Dict[str, Any]:
    result["source"] = source
    result["fallback_reason"] = reason
    return result


COMMON_RULES = """Rules:
- Use INR (₹) prices and products/services that are really available in India (Indian brands where possible).
- "estimated_price" is the price of ONE unit in INR; "quantity" is the number of units.
  The sum of estimated_price x quantity across ALL items must NOT exceed the total budget.
- "search_terms" is a short keyword phrase (2-6 words) that finds the item on Indian shopping sites.
- Reply with valid JSON only. No markdown, no commentary."""

# --------------------------------------------------------------------------- #
# HOME PLANNER
# --------------------------------------------------------------------------- #
HOME_SCHEMA = """{
  "total_budget": 0.0,
  "budget_breakdown": [
    {"category": "lighting | ceiling_fans | furniture | dining_tables | decor",
     "allocation": 0.0,
     "items": [{"name": "", "description": "", "estimated_price": 0.0, "quantity": 1, "search_terms": ""}]}
  ],
  "remaining_budget": 0.0,
  "additional_suggestions": [""]
}"""


def build_home_prompt(inp: HomeBudgetInput) -> str:
    rooms = [n for n, f in (("Living room", inp.has_living_room), ("Kitchen", inp.has_kitchen),
                            ("Bedroom", inp.has_bedroom)) if f]
    return f"""You are an interior-design shopping assistant for the Indian market.
Total budget: ₹{inp.total_budget:,.2f}
Needed:
- {inp.num_lights} lights / lighting fixtures
- {inp.num_fans} ceiling fans
- {inp.num_furniture} furniture pieces
- {inp.num_dining_tables} dining tables
Rooms: {", ".join(rooms) or "not specified"}
Additional requirements: {inp.additional_requirements or "None"}

Balance functionality, style and price. Only include categories the user asked for.
{COMMON_RULES}

Return JSON with exactly this structure:
{HOME_SCHEMA}"""


def _home_fallback(inp: HomeBudgetInput) -> Dict[str, Any]:
    plan = [
        ("lighting", "LED ceiling / panel light (warm white)", "Energy-efficient LED light fixture",
         inp.num_lights, "led ceiling light warm white", 0.20),
        ("ceiling_fans", "Energy-saving ceiling fan (1200 mm)", "BEE star-rated fan with remote",
         inp.num_fans, "bldc ceiling fan 1200mm", 0.30),
        ("furniture", "Furniture piece (engineered wood)", "Durable, budget-friendly furniture",
         inp.num_furniture, "engineered wood furniture", 0.30),
        ("dining_tables", "Dining table set", "Compact dining table with chairs",
         inp.num_dining_tables, "dining table set 4 seater", 0.20),
    ]
    active = [p for p in plan if p[3] > 0] or [
        ("decor", "Home decor bundle", "Wall art, cushions and planters", 1, "home decor items", 1.0)]
    weight = sum(p[5] for p in active)
    cats = []
    for key, name, desc, qty, terms, w in active:
        share = inp.total_budget * 0.9 * w / weight  # keep ~10 % unspent as a buffer
        cats.append({"category": key, "items": [{
            "name": name, "description": desc, "estimated_price": round(share / qty, 2),
            "quantity": qty, "search_terms": terms}]})
    return {"budget_breakdown": cats, "additional_suggestions": [
        "Compare prices across platforms before buying.",
        "Watch for festival sales to save more.",
        "About 10% of the budget is left unspent as a buffer for delivery and installation."]}


def get_home_recommendations(inp: HomeBudgetInput) -> Dict[str, Any]:
    raw, source, reason = _generate(build_home_prompt(inp), lambda: _home_fallback(inp))
    if source == "gemini" and not _breakdown_has_items(raw):
        raw, source, reason = _home_fallback(inp), "fallback", "Gemini returned no usable items."
    result = _finalize_breakdown(raw, inp.total_budget)
    _add_links(result, lambda _k: HOME_PLATFORMS)
    return _tag(result, source, reason)


# --------------------------------------------------------------------------- #
# PARTY PLANNER
# --------------------------------------------------------------------------- #
PARTY_SCHEMA = """{
  "total_budget": 0.0,
  "budget_breakdown": [
    {"category": "venue | catering | decoration | entertainment | contingency",
     "allocation": 0.0,
     "items": [{"name": "", "description": "", "estimated_price": 0.0, "quantity": 1, "search_terms": ""}]}
  ],
  "venue_suggestions": [{"name": "", "type": "", "capacity": 0, "estimated_cost": 0.0, "search_terms": ""}],
  "remaining_budget": 0.0,
  "additional_suggestions": [""]
}"""


def build_party_prompt(inp: PartyBudgetInput) -> str:
    yn = lambda b: "Yes" if b else "No"  # noqa: E731
    return f"""You are a party-planning assistant for India.
Total budget: ₹{inp.total_budget:,.2f}
Party type: {inp.party_type}
Guests: {inp.num_guests}
Venue type: {inp.venue_type or "Not specified"}
Catering needed: {yn(inp.needs_catering)}
Decoration needed: {yn(inp.needs_decoration)}
Entertainment needed: {yn(inp.needs_entertainment)}
Additional requirements: {inp.additional_requirements or "None"}

Split the budget sensibly across venue, catering, decoration, entertainment and a small contingency.
Only include categories the user needs (always keep a contingency of about 5-10%).
For catering use per-plate pricing (quantity = number of guests). Food ideas can be sourced from Swiggy/Zomato.
{COMMON_RULES}

Return JSON with exactly this structure:
{PARTY_SCHEMA}"""


def _party_fallback(inp: PartyBudgetInput) -> Dict[str, Any]:
    at_home = (inp.venue_type or "home").strip().lower() == "home"
    weights = {
        "venue": 0 if at_home else 25,
        "catering": 45 if inp.needs_catering else 0,
        "decoration": 20 if inp.needs_decoration else 0,
        "entertainment": 15 if inp.needs_entertainment else 0,
        "contingency": 10,
    }
    total_w = sum(weights.values())
    amt = {k: inp.total_budget * 0.95 * w / total_w for k, w in weights.items()}
    pt, g = inp.party_type, inp.num_guests
    cats: List[Dict[str, Any]] = []
    if at_home:
        cats.append({"category": "venue", "items": [{
            "name": "Home (no venue cost)", "description": "Host at home", "estimated_price": 0,
            "quantity": 1, "search_terms": ""}]})
    else:
        cats.append({"category": "venue", "items": [{
            "name": f"{inp.venue_type} booking", "description": f"Venue for {g} guests",
            "estimated_price": round(amt["venue"], 2), "quantity": 1,
            "search_terms": f"{inp.venue_type} for {pt} party {g} guests"}]})
    if inp.needs_catering:
        cats.append({"category": "catering", "items": [{
            "name": "Catering per plate", "description": f"Food for {g} guests",
            "estimated_price": round(amt["catering"] / g, 2), "quantity": g,
            "search_terms": f"{pt} party catering"}]})
    if inp.needs_decoration:
        cats.append({"category": "decoration", "items": [{
            "name": f"{pt} decoration kit", "description": "Balloons, banners, lights",
            "estimated_price": round(amt["decoration"], 2), "quantity": 1,
            "search_terms": f"{pt} party decoration kit"}]})
    if inp.needs_entertainment:
        cats.append({"category": "entertainment", "items": [{
            "name": "Games and music", "description": "Party games, speaker or DJ playlist",
            "estimated_price": round(amt["entertainment"], 2), "quantity": 1,
            "search_terms": f"{pt} party games"}]})
    cats.append({"category": "contingency", "items": [{
        "name": "Unexpected expenses", "description": "Buffer for last-minute costs",
        "estimated_price": round(amt["contingency"], 2), "quantity": 1, "search_terms": ""}]})
    venues = [] if at_home else [{
        "name": f"{inp.venue_type} for {g} guests", "type": inp.venue_type, "capacity": g,
        "estimated_cost": round(amt["venue"], 2), "search_terms": f"{inp.venue_type} party venue {g} guests"}]
    return {"budget_breakdown": cats, "venue_suggestions": venues, "additional_suggestions": [
        "Consider a potluck-style menu to reduce catering costs.",
        "Book venue and vendors early for better rates.",
        "Homemade decorations are a cheap alternative."]}


def get_party_recommendations(inp: PartyBudgetInput) -> Dict[str, Any]:
    raw, source, reason = _generate(build_party_prompt(inp), lambda: _party_fallback(inp))
    if source == "gemini" and not _breakdown_has_items(raw):
        raw, source, reason = _party_fallback(inp), "fallback", "Gemini returned no usable items."
    result = _finalize_breakdown(raw, inp.total_budget)
    _add_links(result, lambda key: PARTY_PLATFORMS.get(key, DEFAULT_PARTY_PLATFORMS))

    venues = []
    for v in raw.get("venue_suggestions") or []:
        if not isinstance(v, dict):
            continue
        v = dict(v)
        v["name"] = str(v.get("name") or "Venue")
        v["type"] = str(v.get("type") or "")
        v["capacity"] = int(_num(v.get("capacity")))
        v["estimated_cost"] = max(_num(v.get("estimated_cost")), 0.0)
        terms = str(v.get("search_terms") or v["name"])
        v["search_links"] = build_links(VENUE_PLATFORMS, terms)
        venues.append(v)
    result["venue_suggestions"] = venues
    return _tag(result, source, reason)


# --------------------------------------------------------------------------- #
# JEWELRY PLANNER (text + optional outfit image)
# --------------------------------------------------------------------------- #
JEWELRY_SCHEMA = """{
  "outfit_analysis": {"colors": [""], "style": "", "formality": ""},
  "total_budget": 0.0,
  "jewelry_recommendations": [
    {"item_type": "", "description": "", "style": "", "estimated_price": 0.0, "search_terms": ""}
  ],
  "remaining_budget": 0.0,
  "styling_tips": [""]
}"""


def build_jewelry_prompt(inp: JewelryBudgetInput, has_image: bool) -> str:
    image_note = ("An image of the outfit is attached. Analyse its colours, style and formality, then suggest "
                  "jewelry that complements it. Fill in \"outfit_analysis\"."
                  if has_image else
                  "No outfit image was provided; omit \"outfit_analysis\" from the JSON.")
    return f"""You are a jewelry stylist for the Indian market.
Total budget: ₹{inp.total_budget:,.2f}
Occasion: {inp.occasion}
Style preferences: {inp.preferences or "Not specified"}
{image_note}

Suggest 3-5 pieces that fit together and suit the occasion. Each item is a single piece (quantity is always 1).
Rules:
- Use INR (₹) prices for jewelry available in India (Indian brands / platforms).
- The sum of all estimated_price values must NOT exceed the total budget.
- "search_terms" is a short keyword phrase that finds the item on Indian shopping sites.
- Reply with valid JSON only. No markdown, no commentary.

Return JSON with exactly this structure:
{JEWELRY_SCHEMA}"""


def _jewelry_fallback(inp: JewelryBudgetInput) -> Dict[str, Any]:
    prefs = (inp.preferences or "").strip()
    plan = [("necklace", 0.35), ("earrings", 0.25), ("bracelet", 0.20), ("ring", 0.10)]
    items = []
    for kind, w in plan:
        items.append({
            "item_type": kind,
            "description": f"A {kind} that suits a {inp.occasion.lower()} look.",
            "style": prefs.split(",")[0][:30] if prefs else "classic",
            "estimated_price": round(inp.total_budget * w, 2),
            "search_terms": f"{prefs.split(',')[0] + ' ' if prefs else ''}{kind} for {inp.occasion}".strip(),
        })
    return {"jewelry_recommendations": items, "styling_tips": [
        "Keep one statement piece and let the rest stay simple.",
        "Match metal tones (gold, silver or rose gold) across pieces.",
        "Add an outfit photo with a Gemini API key for colour-matched suggestions."]}


def get_jewelry_recommendations(inp: JewelryBudgetInput, image_bytes: Optional[bytes] = None,
                                mime_type: Optional[str] = None) -> Dict[str, Any]:
    prompt = build_jewelry_prompt(inp, bool(image_bytes))
    raw, source, reason = _generate(prompt, lambda: _jewelry_fallback(inp), image_bytes, mime_type)
    items = _prepare_items(raw.get("jewelry_recommendations"))
    if source == "gemini" and not items:
        raw, source, reason = _jewelry_fallback(inp), "fallback", "Gemini returned no usable items."
        items = _prepare_items(raw["jewelry_recommendations"])

    notes = _fit_to_budget(items, inp.total_budget)
    spent = 0.0
    for i in items:
        i["item_type"] = str(i.get("item_type") or i["name"])
        i["style"] = str(i.get("style") or "")
        i["line_total"] = round(i["estimated_price"], 2)
        spent += i["line_total"]
        i["shopping_links"] = build_links(JEWELRY_PLATFORMS, i["search_terms"] or i["item_type"])

    result: Dict[str, Any] = {
        "total_budget": round(inp.total_budget, 2),
        "jewelry_recommendations": items,
        "total_spent": round(spent, 2),
        "remaining_budget": round(inp.total_budget - spent, 2),
        "styling_tips": _str_list(raw.get("styling_tips")) + notes,
    }
    analysis = raw.get("outfit_analysis")
    if image_bytes and isinstance(analysis, dict):
        result["outfit_analysis"] = {
            "colors": _str_list(analysis.get("colors")),
            "style": str(analysis.get("style") or ""),
            "formality": str(analysis.get("formality") or ""),
        }
    return _tag(result, source, reason)