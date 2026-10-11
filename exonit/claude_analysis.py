"""Claude API integration for intelligent candidate analysis.

Uses Anthropic Tool Use API for structured JSON output — no regex parsing.

Sends strong transit candidates to Claude for analysis including:
- Cross-referencing with known exoplanet catalogs
- Evaluating transit parameters for physical plausibility
- Generating follow-up recommendations
- Preparing cTOI submission summaries
"""

import os
import json


# Tool definition for structured output
_ANALYSIS_TOOL = {
    "name": "submit_analysis",
    "description": "Submit the transit candidate analysis results.",
    "input_schema": {
        "type": "object",
        "properties": {
            "verdict": {
                "type": "string",
                "enum": ["planet_candidate", "likely_false_positive", "uncertain"],
                "description": "Overall assessment of the candidate.",
            },
            "confidence": {
                "type": "number",
                "description": "Confidence in the verdict (0.0 to 1.0).",
            },
            "analysis": {
                "type": "string",
                "description": "Brief scientific analysis (2-3 sentences).",
            },
            "physical_plausibility": {
                "type": "string",
                "description": "Assessment of transit parameters considering stellar type.",
            },
            "known_matches": {
                "type": "string",
                "description": "Any known planets/candidates at similar coordinates/period.",
            },
            "follow_up": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Recommended follow-up observations.",
            },
            "ctoi_summary": {
                "type": ["string", "null"],
                "description": "One-paragraph summary for cTOI submission, or null if not warranted.",
            },
        },
        "required": ["verdict", "confidence", "analysis"],
    },
}


CLAUDE_MODEL = "claude-haiku-4-5-20251001"  # default: cheapest
CLAUDE_MODEL_STRONG = "claude-sonnet-4-6"  # for top candidates


def analyze_candidate(candidate, vetting_result, star_info=None,
                      model=None, api_key=None):
    """Analyze a transit candidate using Claude API.

    Args:
        candidate: Dict with period_days, depth_ppm, snr, t0, duration_days, etc.
        vetting_result: Dict with statistical vetting flags.
        star_info: Optional dict with stellar parameters (tic_id, tmag, ra, dec).
        model: Claude model to use (default: haiku for cost efficiency).
        api_key: Anthropic API key (default: from env).

    Returns:
        Dict with verdict, analysis, follow_up, ctoi_summary, or None on error.
    """
    try:
        import anthropic
    except ImportError:
        print("  Warning: anthropic SDK not installed. Skipping AI analysis.")
        return None

    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        print("  Warning: ANTHROPIC_API_KEY not set. Skipping AI analysis.")
        return None

    model = model or CLAUDE_MODEL
    client = anthropic.Anthropic(api_key=key)

    # Build prompt
    prompt = _build_analysis_prompt(candidate, vetting_result, star_info)

    try:
        response = client.messages.create(
            model=model,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
            system="You are an expert exoplanet astronomer reviewing transit candidates from TESS data. Be concise and scientific. Use the submit_analysis tool to report your findings.",
            tools=[_ANALYSIS_TOOL],
            tool_choice={"type": "tool", "name": "submit_analysis"},
        )

        # Extract structured result from tool use block
        for block in response.content:
            if block.type == "tool_use" and block.name == "submit_analysis":
                result = block.input
                result["model_used"] = model
                return result

        # Fallback: no tool use block (shouldn't happen with tool_choice)
        print(f"  Warning: Claude did not use submit_analysis tool")
        return {"verdict": "error", "analysis": "No tool use in response",
                "confidence": 0, "model_used": model}

    except Exception as e:
        print(f"  Claude API error: {e}")
        return None


def _build_analysis_prompt(candidate, vetting, star_info):
    """Build the analysis prompt for Claude."""
    star_id = (star_info or {}).get("tic_id", "unknown")
    tmag = (star_info or {}).get("tmag", "unknown")
    ra = (star_info or {}).get("ra", "unknown")
    dec = (star_info or {}).get("dec", "unknown")

    stellar_radius = (star_info or {}).get("stellar_radius", "unknown")
    teff = (star_info or {}).get("teff", "unknown")
    logg = (star_info or {}).get("logg", "unknown")

    return f"""Analyze this TESS transit candidate.

## Star
- TIC ID: {star_id}
- TESS magnitude: {tmag}
- Coordinates: RA={ra}, Dec={dec}
- Stellar radius: {stellar_radius} R_sun
- Effective temperature: {teff} K
- Surface gravity (log g): {logg}

## Important Context
- TESS observes many M-dwarf stars (small stars, R* = 0.1-0.6 R_sun)
- For M-dwarfs, transit depths of 500-2000+ ppm are NORMAL for Earth/super-Earth sized planets
- Do NOT flag a candidate as false positive solely based on transit depth without considering stellar radius
- A 1000 ppm depth around an M-dwarf (R*=0.4 R_sun) implies Rp ~ 1.3 R_earth — perfectly plausible

## Candidate
- Period: {candidate.get('period_days', 'N/A')} days
- Transit depth: {candidate.get('depth_ppm', 'N/A')} ppm
- Duration: {candidate.get('duration_days', 'N/A')} days
- SNR (SDE): {candidate.get('snr', 'N/A')}
- Expected transits: {candidate.get('n_transits_expected', 'N/A')}
- CNN planet score: {candidate.get('cnn_score', 'N/A')} (0=FP, 1=planet)

## Statistical Vetting
- Harmonics check: {not vetting.get('is_harmonic', True)}
- Odd/even consistent: {vetting.get('odd_even_consistent', 'N/A')}
- V-shape (binary): {vetting.get('is_v_shaped', 'N/A')}
- Overall: {'PASSED' if vetting.get('passed') else 'FAILED'}

Use the submit_analysis tool with your assessment."""


def should_analyze(candidate, vetting_result, cnn_score=None,
                   snr_threshold=10.0, cnn_threshold=0.7):
    """Decide whether a candidate warrants Claude API analysis.

    Only sends candidates that:
    - Pass statistical vetting
    - Have SNR > threshold
    - Have CNN score > threshold (if available)

    Args:
        candidate: Candidate dict.
        vetting_result: Vetting result dict.
        cnn_score: Optional CNN classification score.
        snr_threshold: Minimum SNR for API analysis.
        cnn_threshold: Minimum CNN score for API analysis.

    Returns:
        bool: Whether to send to Claude API.
    """
    if not vetting_result.get("passed", False):
        return False
    if candidate.get("snr", 0) < snr_threshold:
        return False
    if cnn_score is not None and cnn_score < cnn_threshold:
        return False
    return True


def select_model(candidate):
    """Select Claude model based on candidate strength.

    Uses cheaper model for routine analysis, stronger model for
    exceptional candidates.
    """
    if candidate.get("snr", 0) > 15:
        return CLAUDE_MODEL_STRONG
    return CLAUDE_MODEL
