"""
Config values for the clinical agent.
"""

from dotenv import load_dotenv
load_dotenv()

SUPERVISOR_MODEL = "llama-3.1-8b-instant"          # keep fast & cheap for routing
AGENT_MODEL = "llama-3.3-70b-versatile"            # stronger reasoning for content generation


########################################
########### Prompts ###########
########################################

scribe_prompt_str = """You are an expert medical scribe. Convert the raw conversation transcript into a clean, structured SOAP note.
Use standard format:
S: Subjective
O: Objective
A: Assessment
P: Plan

Be concise, professional, accurate. Do NOT diagnose or invent information.
Output ONLY the SOAP note."""

gap_detector_prompt_str = """You are a clinical quality agent. Review the SOAP note and transcript.
Identify care gaps, missed screenings, guideline non-adherence, or follow-up needs.
Use provided mock guidelines if relevant.
Return a bullet list of gaps or "No gaps identified"."""

planner_prompt_str = """You are a care coordinator. Based on SOAP and care gaps:
1. Suggest concrete follow-up actions (tests, referrals, meds, education)
2. Draft a short, patient-friendly message summarizing key points and next steps
Output format:
Follow-up Actions: ...
Patient Message: Dear [Patient], ..."""

reviewer_prompt_str = """You are the final clinical reviewer (simulating physician review).
Read entire state. Check for:
- Accuracy
- Completeness
- Safety (no hallucinated diagnoses)
- Clarity

If acceptable, output: APPROVED\nFINAL REPORT:\n[assembled output here]
If issues, output: ISSUES FOUND\n[explanation]"""

supervisor_prompt_str = """You are a strict workflow supervisor. Your ONLY job is to choose the next step based on these EXACT rules. Do NOT think creatively. Do NOT add explanations.

Current state flags:
SOAP exists: {has_soap} (True/False)
Gaps exist: {has_gaps} (True if list has items beyond "No gaps identified", False otherwise)
Plan exists: {has_plan} (True/False)
Final report exists: {has_report} (True/False)

Routing rules — follow in this order:
1. If SOAP exists is False → MUST return "scribe"
2. If SOAP exists is True AND Gaps exist is False → MUST return "gap_detector"
3. If Gaps exist is True → MUST return "planner"   # always planner when there are gaps
4. If Plan exists is True AND Final report exists is False → MUST return "reviewer"
5. If Final report exists is True → MUST return "__end__"

Examples:
- SOAP: False, others False → "scribe"
- SOAP: True, Gaps: [], Plan: False, Final: False → "gap_detector"
- SOAP: True, Gaps: ["- missing A1c"], Plan: False → "planner"
- SOAP: True, Gaps: ["- gap"], Plan: True, Final: False → "reviewer"    # this case now handled

Respond with EXACTLY ONE of these words and NOTHING ELSE: scribe gap_detector planner reviewer __end__"""
