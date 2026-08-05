#!/usr/bin/env python3
# loosely based on https://github.com/openai/emoclassifiers, heavily adapted

import argparse
import csv
import glob
import json
import os
import re
import sys
import pydantic
from difflib import SequenceMatcher
from enum import Enum
from typing import List, Optional, Type
from openai import OpenAI, BadRequestError


# ==========================
# CONFIG
# ==========================

PROVIDER          = "openrouter"   # or "openai"
INPUT_DIR         = "input"
OUTPUT_CSV        = "codes.csv"          # session_id, chunk_id, result
STAGE2_INPUT_CSV  = "inputfeatures.csv"  # stage-02 input
SOLVES_CSV        = None                 # optional: session_id,solved

MODEL_OPENAI     = "gpt-5.2-2025-12-11"
MODEL_OPENROUTER = "openai/gpt-5.2"
SEED = 1337 ; REASONING_EFFORT = "minimal" ; TEMPERATURE = 0.0 ; VERBOSITY = "low"
MAX_OUTPUT_TOKENS = 1000


# ==========================
# RESPONSE FORMAT
# ==========================

class CodeEnum(str, Enum):
    DEBUGGING = 'debugging'
    CONFIRMATION = 'confirmation'
    PASTED_CONTEXT_DUMP = 'pasted_context_dump'
    OBSERVATIONS = 'observations'
    REQUEST_DIRECTION = 'request_direction'
    HELP_REQUEST = 'help_request'
    CONFUSION = 'confusion'
    VAGUE_REQUEST = 'vague_request'
    CODE_GENERATION = 'code_generation'
    CONCEPT_GUIDANCE_PROCEDURAL = 'concept_guidance_procedural'
    CONCEPT_GUIDANCE_NON_PROCEDURAL = 'concept_guidance_non_procedural'
    CHALLENGE_UNDERSTANDING = 'challenge_understanding'
    DIRECT_SOLUTION = 'direct_solution'
    COURSE_PLATFORM = 'course_platform'
    SOCIAL_TURN = 'social_turn'
    NON_SEQUITUR = 'non_sequitur'


class ResponseFormat(pydantic.BaseModel):
    result: CodeEnum
    justification: str
    confidence: int


CLASSIFIER_PROMPT = """
You are an expert qualitative coder responsible for classifying messages from students to a cybersecurity tutoring system, SENSAI.

<output_spec>
Return **only** valid JSON with these exact keys:
{{
  "result": one of {{debugging | confirmation | pasted_context_dump | observations | request_direction | help_request | confusion | vague_request | concept_guidance_procedural | concept_guidance_non_procedural | code_generation | challenge_understanding | direct_solution | social_turn | non_sequitur | course_platform}},
  "confidence": integer 1-5,
  "justification": string (20-100 words, format below)
}}

**Confidence scale:**
- **5**: Explicit match to heuristics, zero ambiguity
- **4**: Clear match, minor ambiguity in phrasing
- **3**: Reasonable fit but required some interpretation
- **2**: Borderline case, could plausibly be another code
- **1**: Required significant inference, flag for review

**Justification format (MANDATORY):**
1. Start with ‹paraphrased learner phrase, max 12 words, from marked message ONLY›
2. Then 20-100 words explaining:
   - WHY the chosen category fits
   - WHY the most likely alternative(s) do NOT fit (prefix rejected codes with "not_")

Example: ‹how would I send a GET request› asks for procedural steps. Not_request_direction because contains specific action verb; not_code_generation because no explicit code request.
</output_spec>

<anti_inference_rules>
**CRITICAL: Code only what is OBSERVABLE in the marked message.**

FORBIDDEN in justifications:
- "implying", "implies", "suggesting", "suggests"
- "likely", "probably", "presumably", "apparently"
- "seems to", "appears to", "might be", "could be"
- "they want to", "they are trying to", "their intent is"
- Any reasoning about what the learner is "really" doing or thinking

REQUIRED: Base classification ONLY on:
- Explicit words and phrases present in the message
- Syntactic patterns (question marks, modal verbs, "how/what/why")
- Presence/absence of error output, code, natural language
- Match against heuristic ✓ and ✘ criteria

If you cannot classify without inference, set confidence to 1 or 2 and classify based on the MOST observable pattern. Do not guess at hidden intent.

BAD: "implying they are running into execution issues" (inference about state)
GOOD: "contains code snippet with no explicit question" (observable pattern)

BAD: "likely tied to unexpected behavior" (guessing cause)
GOOD: "no error output visible in message" (observable fact)
</anti_inference_rules>

<verbosity_constraint>
- Do NOT add commentary outside the JSON
- Do NOT explain your reasoning before the JSON
- Output ONLY the JSON object
</verbosity_constraint>

### DECISION ORDER (stop at first match) ###

Process in this exact order. Stop at the FIRST matching category.

<early_stop_criteria>
- Once a category's heuristics (✓) are satisfied, STOP immediately
- Do NOT continue checking lower-priority categories
- Do NOT second-guess a match by exploring alternatives
- The decision order IS the plan—execute it, don't reason beyond it
</early_stop_criteria>

**Priority 1: Platform/Logistics**
→ course_platform

**Priority 2: Solution Outsourcing**
→ direct_solution

**Priority 3: Error Output OR Failure Report**
→ debugging

**Priority 4: Validation/Confirmation Patterns**
→ confirmation

**Priority 5: Pure Artifact (no human framing)**
→ pasted_context_dump

**Priority 6: Code Writing Request**
→ code_generation

**Priority 7: Conceptual/Procedural Questions**
→ concept_guidance_procedural OR concept_guidance_non_procedural

**Priority 8: Challenge-Specific Understanding**
→ challenge_understanding

**Priority 9: Direction/Navigation**
→ request_direction

**Priority 10: Explicit Confusion**
→ confusion

**Priority 11: Generic Help**
→ help_request

**Priority 12: Factual Observations**
→ observations

**Priority 13: Social/Phatic**
→ social_turn

**Priority 14: Underspecified Request**
→ vague_request

**Priority 15: Fallback**
→ non_sequitur

---

### COMPLETE CODEBOOK ###

#### fixing_verifying (Priority 3-4)

**debugging**
- Learner reports a problem with THEIR work and implicitly/explicitly expects help fixing it
- Includes: error output, failure reports, "not working" statements
- KEY INSIGHT: When learner tells the tutor "there is an issue", they expect the tutor to help fix it

**Triggers for debugging:**
✓ True error output: "Permission denied", "No such file", "404", "invalid", "failed", "error:", stack traces
✓ Failure statements: "doesn't work", "not working", "didn't work", "still not working"
✓ Diagnosis questions: "what's wrong?", "why isn't this working?", "what am I doing wrong?"
✓ Problem reports: "crashing", "failing", "broken", "stuck at [technical state]"
✓ "that didn't work either" (reporting continued failure)

**NOT errors (informational output):**
✗ File type info, process IDs, scan results, timestamps
✗ Normal command output showing what ran successfully
✗ Progress messages ("Starting...", "Completed...", "Scanning...")

**Question type can OVERRIDE to other categories:**
- "is my X wrong?" / "is it because X?" → confirmation (yes/no validation)
- Context + "how do I X?" → concept_guidance (classify the question)
- Context + "what should I do?" → request_direction (classify the question)

**Attribution to SENSAI goes to observations:**
✗ "your code doesn't work" / "I copied what you gave me" → observations (feedback about SENSAI, not their problem)

Decision heuristics:
✓ Error output present in learner message
✓ Failure language: "doesn't work", "not working", "failed", "broken"
✓ Diagnosis-seeking: "what's wrong", "why doesn't this"
✘ Yes/no question about correctness: "is X wrong?" → confirmation
✘ Attribution to SENSAI: "your suggestion didn't work" → observations
✘ Statement of confused STATE: "I don't know what I'm doing wrong" → confusion

**confirmation**
- Asking for validation of an approach, method, code, or understanding
- Includes yes/no questions about correctness or cause
- Decision heuristics:
  ✓ Modal/validation patterns: "can I", "should I", "do I need to", "would it work to", "is this correct"
  ✓ Correctness questions: "is my X wrong?", "is this right?"
  ✓ Cause validation: "is it because X?", "is that why Y?"
  ✓ Showing code/key/output and asking for validation
  ✓ Tag-question patterns: "right?", "correct?", "yeah?"
  ✘ NOT open "how" questions → concept_guidance
  ✘ NOT failure statements "it doesn't work" → debugging
  ✘ NOT statements of confused state → confusion

---

#### providing_info (Priority 5, 12)

**pasted_context_dump**
- ENTIRE message is pasted artifact with NO human framing or intent
- Artifacts: terminal commands/output, source code, logs, config files, hex/binary data, URLs, file paths
- Decision heuristics:
  ✓ Could be pasted directly into code file, shell, REPL, URL bar
  ✓ No natural-language sentences or clauses
  ✓ Tiny trailing tokens like lone "?" don't break purity
  ✘ ANY natural-language commentary ("this is what I got", "here is my code") → observations
  ✘ True error output present → debugging
  ✘ Very short ambiguous word with no code structure → non_sequitur

**observations**
- Factual statements or status updates; feedback about SENSAI's suggestions
- Decision heuristics:
  ✓ Describes actions taken, outputs received, status
  ✓ "I am doing it in python", "it gave me X", "I don't see anything"
  ✓ Attribution to SENSAI: "your code doesn't work", "I copied what you gave me", "you forgot the read"
  ✓ Introduces artifact WITH commentary ("this is what my terminal shows:", "here is my code:")
  ✓ Neutral corrections: "no not correct", "that's not right"
  ✓ Contains at least some clausal structure (subject + verb)
  ✘ Reports own failure without SENSAI attribution → debugging
  ✘ Seeks validation ("is this okay?") → confirmation
  ✘ Explicitly states confusion → confusion

---

#### getting_unstuck (Priority 9-11, 14)

**request_direction**
- Asks for next step/direction WITHOUT asking HOW to do something specific
- Decision heuristics:
  ✓ "what should I do next?", "what am I supposed to do?", "where should I look?"
  ✓ Pure sequencing/navigation with NO technical verb after "how/what/where"
  ✓ "how do I solve this challenge" (meta-question about approach, not technique)
  ✓ "how do I proceed", "how should I approach this"
  ✘ NEVER if contains "how do I" + specific action verb → concept_guidance
  ✘ NEVER if asks "what command", "what tool" → concept_guidance

**help_request**
- Generic help request with NO specifics
- MUST explicitly contain the word "help"
- Decision heuristics:
  ✓ "Can you help me with this challenge?"
  ✓ "I need help", "please help"
  ✘ Any additional technical detail → concept_guidance or debugging

**confusion**
- Explicitly states being confused, lost, or stuck as a STATE
- KEY: Statements about their mental state, not diagnosis-seeking
- Decision heuristics:
  ✓ Contains: "confused", "not sure", "no idea", "lost", "stuck", "idk"
  ✓ "I don't know what I'm doing wrong" (statement of state)
  ✓ "I don't understand what this challenge is asking"
  ✓ "that doesn't make sense"
  ✘ Active diagnosis question "what am I doing wrong?" with shown work/error → debugging
  ✘ Reports technical failure → debugging

**vague_request**
- Very short, underspecified request for more/different from SENSAI
- Decision heuristics:
  ✓ "more", "more?", "how?", "then?", "check again"
  ✓ "give me instructions" (underspecified, can't assume wants solution)
  ✓ "explain more please"
  ✓ Short imperatives with only vague pronoun
  ✘ Explicit hint/solution request → direct_solution
  ✘ Factual update → observations
  ✘ Greeting/thanks → social_turn

---

#### implementing (Priority 6-7a)

**code_generation**
- EXPLICIT request for SENSAI to write code or template code
- Decision heuristics:
  ✓ "can you give me a code template", "write a script to", "give me example code"
  ✓ "give me the syntax" (asking for code form)
  ✓ Asks for code snippets (not complete solutions)
  ✘ Asking how to do something → concept_guidance_procedural
  ✘ Asking for the full solution → direct_solution

**concept_guidance_procedural**
- HOW-TO questions: wants steps, commands, code, concrete actions to perform
- Decision heuristics:
  ✓ "how do I [action verb]", "how to [action]", "show me how to"
  ✓ "how do I compile this", "how do I run this"
  ✓ "what is a way to [achieve X]" expecting concrete technique
  ✓ Verbs: block, find, remove, initialize, intercept, send, read, write, modify, exploit, compile, run
  ✘ Modal validation ("should I", "can I" + step) → confirmation
  ✘ Asking for criteria/rules/reasons → concept_guidance_non_procedural
  ✘ Meta "how do I solve/approach this challenge" → request_direction

---

#### understanding (Priority 7b-8)

**concept_guidance_non_procedural**
- WHAT/WHY/EXPLAIN questions: wants meaning, reasons, criteria, categories
- Decision heuristics:
  ✓ "what does this mean", "why do I need to", "explain [concept]"
  ✓ "how do I know which..." (asking for criteria, not procedure)
  ✓ "what kind of", "what type of"
  ✓ "why do I get e if I don't use it" (asking about RSA concepts - generalizable)
  ✓ Mixed "what + how" where conceptual "what/why" is present → non_procedural wins
  ✘ Clearly wants steps to execute → concept_guidance_procedural

**challenge_understanding**
- Questions ONLY specific to understanding the CURRENT challenge setup
- Decision heuristics:
  ✓ "what IP/url/port/file do I use for this challenge"
  ✓ "which user should I login as"
  ✓ "how is this challenge different from [other challenge]"
  ✓ "how do I get the color data for this challenge" (challenge-specific artifact)
  ✓ Information is NOT generalizable beyond this challenge
  ✘ Generalizable knowledge (RSA math, setuid semantics) → concept_guidance
  ✘ "how do I solve this" → request_direction

---

#### outsource (Priority 2)

**direct_solution**
- Requests the FULL solution OR explicit hint/shortcut to the answer
- Decision heuristics:
  ✓ "give me the full script", "just give me the answer"
  ✓ "give me a hint", "can I get a hint", "hint please" (seeking shortcut)
  ✓ "give me the code", "give me the code to run" (generic code request = solution)
  ✓ "steps to follow in this question" (wants walkthrough)
  ✓ "tell me the answer"
  ✘ "give me instructions" → vague_request (not explicit solution request)
  ✘ Template/example code for learning → code_generation

---

#### peripheral (Priority 1, 13, 15)

**course_platform**
- Questions about course logistics, platform functionality, grading
- Decision heuristics:
  ✓ "how can I request an extension", "how do I submit"
  ✓ "how many characters is the flag usually" (flag format = platform knowledge)
  ✘ Technical questions about the challenge → other codes

**social_turn**
- Social or phatic messages: greetings, acknowledgments, thanks
- Decision heuristics:
  ✓ "got it", "yes", "thanks", "hello", "ohhh", "yes please"
  ✓ "it worked", "thank you"
  ✓ Very short (1-5 words), serves acknowledgment/politeness
  ✘ Corrections/feedback about SENSAI → observations
  ✘ Single ambiguous word → non_sequitur

**non_sequitur**
- Too vague, fragmented, or out of context to interpret
- Decision heuristics:
  ✓ Typos, incomplete thoughts, bare tokens
  ✓ Single ambiguous words: "flag", "lets analyze"
  ✓ Nonsense or off-topic content
  ✘ Clear request word ("more", "hint") → vague_request or direct_solution
  ✘ Code/command/URL → pasted_context_dump

---

### EDGE CASE CLARIFICATIONS ###

1. **"is my X wrong?" vs "X is wrong/not working":**
   - "is my file path wrong?" → confirmation (yes/no question)
   - "my file path is wrong" / "it's not working" → debugging (failure report)

2. **Failure attribution:**
   - "it doesn't work" → debugging (their problem)
   - "your code doesn't work" / "I copied what you gave me and it fails" → observations (SENSAI attribution)
   - "that didn't work either" → debugging (reporting continued failure of their attempt)

3. **Confusion vs Debugging:**
   - "I don't know what I'm doing wrong" → confusion (statement of state)
   - "what am I doing wrong?" + shown work/error → debugging (diagnosis-seeking)
   - "I am stuck" → confusion

4. **Hint requests:**
   - "give me a hint" / "hint please" / "hint" → direct_solution (seeking shortcut)
   - "give me instructions" → vague_request (underspecified)

5. **How do I solve/approach:**
   - "how do I solve this challenge" → request_direction (meta-question)
   - "how do I solve for X" (specific technique) → concept_guidance_procedural

6. **Terminal output classification:**
   - True error in output → debugging
   - Informational output only → pasted_context_dump
   - Output + specific question → classify the question

7. **Single word messages:**
   - "flag" → non_sequitur
   - "hint" → direct_solution
   - "more" → vague_request
   - "thanks" → social_turn

---

### EXAMPLES ###

**debugging:**
[Learner] bash: cd: /files: No such file or directory [/Learner]
{{"result":"debugging","confidence":5,"justification":"‹No such file or directory› shows bash error output. Not_confirmation because not a yes/no question; not_observations because reports own error not SENSAI feedback."}}

[Learner] doesnt work [/Learner]
{{"result":"debugging","confidence":4,"justification":"‹doesnt work› reports failure state expecting tutor help. Not_observations because reports own problem not SENSAI attribution; not_confusion because states failure not confused state."}}

[Learner] that didnt work either [/Learner]
{{"result":"debugging","confidence":4,"justification":"‹didnt work either› reports continued failure after previous attempt. Not_observations because refers to own attempt not SENSAI suggestion; not_vague_request because reports specific failure state."}}

[Learner] its still not working whats wrong? [/Learner]
{{"result":"debugging","confidence":5,"justification":"‹still not working whats wrong› combines failure report with diagnosis question. Not_confusion because actively seeks diagnosis; not_observations because asks for help."}}

**confirmation:**
[Learner] is my file path wrong [/Learner]
{{"result":"confirmation","confidence":5,"justification":"‹is my file path wrong› asks yes/no validation question about correctness. Not_debugging because question format seeks validation not diagnosis; not_concept_guidance because no how-to request."}}

[Learner] is it because that it must be provided with a directory? [/Learner]
{{"result":"confirmation","confidence":5,"justification":"‹is it because› asks yes/no question about cause. Not_debugging because seeks validation of hypothesis; not_concept_guidance because no procedure requested."}}

[Learner] so essentially I have to send a corrupted payload? [/Learner]
{{"result":"confirmation","confidence":5,"justification":"‹so essentially I have to› asks to validate proposed approach. Not_concept_guidance because no instructions requested; not_request_direction because proposes specific method."}}

**observations:**
[Learner] I copied the code from you so I think this is correct [/Learner]
{{"result":"observations","confidence":4,"justification":"‹copied the code from you› attributes work to SENSAI and reports status. Not_confirmation because states belief not asks question; not_debugging because attributes to SENSAI not own problem."}}

[Learner] you forgot about the read [/Learner]
{{"result":"observations","confidence":5,"justification":"‹you forgot› provides feedback about SENSAI's omission. Not_debugging because attributes issue to SENSAI; not_social_turn because has technical content."}}

[Learner] no not correct [/Learner]
{{"result":"observations","confidence":4,"justification":"‹no not correct› provides correction feedback to SENSAI. Not_social_turn because contains evaluative content; not_debugging because corrects SENSAI not reports own failure."}}

**confusion:**
[Learner] i dont know what im doing wrong [/Learner]
{{"result":"confusion","confidence":4,"justification":"‹dont know what im doing wrong› states confused mental state. Not_debugging because expresses state of not-knowing rather than seeking active diagnosis; not_help_request because no explicit help keyword."}}

[Learner] I am stuck [/Learner]
{{"result":"confusion","confidence":5,"justification":"‹I am stuck› explicitly states stuck state. Not_debugging because no failure report or diagnosis request; not_vague_request because describes state not requests action."}}

[Learner] that doesnt make sense [/Learner]
{{"result":"confusion","confidence":4,"justification":"‹doesnt make sense› expresses lack of understanding. Not_debugging because no technical failure reported; not_observations because expresses confusion not factual status."}}

**direct_solution:**
[Learner] give me a hint [/Learner]
{{"result":"direct_solution","confidence":5,"justification":"‹give me a hint› explicitly requests hint/shortcut to solution. Not_vague_request because hint is explicit solution-seeking; not_help_request because requests shortcut not general help."}}

[Learner] Steps to follow in this question [/Learner]
{{"result":"direct_solution","confidence":4,"justification":"‹steps to follow› requests complete walkthrough of solution. Not_request_direction because wants full steps not just next step; not_concept_guidance because wants solution not technique explanation."}}

[Learner] giev me the code to run [/Learner]
{{"result":"direct_solution","confidence":4,"justification":"‹give me the code to run› requests complete working code for solution. Not_code_generation because wants full solution not template; not_concept_guidance because no learning intent."}}

**vague_request:**
[Learner] give me instructions [/Learner]
{{"result":"vague_request","confidence":4,"justification":"‹give me instructions› is underspecified request without clear scope. Not_direct_solution because instructions could mean guidance not full answer; not_request_direction because no specific navigation asked."}}

[Learner] hint [/Learner]
{{"result":"direct_solution","confidence":5,"justification":"‹hint› single word explicitly requests hint shortcut. Not_vague_request because hint has specific meaning of solution shortcut; not_non_sequitur because clear intent."}}

**pasted_context_dump:**
[Learner] mov rax, QWORD PTR [rsp] add rax, QWORD PTR [rsp + 0x08] [/Learner]
{{"result":"pasted_context_dump","confidence":5,"justification":"‹mov rax, QWORD PTR› is pure assembly code with no human framing. Not_observations because no natural language; not_debugging because no error."}}

[Learner] hacker@dojo:~$ nmap -p 31337 10.0.0.0/16
Starting Nmap 7.95 at 2025-02-27
Nmap done: 65536 IP addresses scanned [/Learner]
{{"result":"pasted_context_dump","confidence":5,"justification":"‹nmap... Nmap done› shows informational scan output with no errors. Not_debugging because no error present; not_observations because no human commentary."}}

**request_direction:**
[Learner] how do I solve this challenge [/Learner]
{{"result":"request_direction","confidence":4,"justification":"‹how do I solve this challenge› is meta-question about approach. Not_concept_guidance_procedural because no specific technique asked; not_direct_solution because asks for direction not answer."}}

[Learner] what should I do next [/Learner]
{{"result":"request_direction","confidence":5,"justification":"‹what should I do next› seeks navigation guidance. Not_concept_guidance because no technical concept; not_help_request because no help keyword."}}

**concept_guidance_procedural:**
[Learner] how do I compile this [/Learner]
{{"result":"concept_guidance_procedural","confidence":5,"justification":"‹how do I compile› asks for concrete compilation steps. Not_request_direction because compile is specific action; not_code_generation because asks how-to not code output."}}

[Learner] how would I send a GET request [/Learner]
{{"result":"concept_guidance_procedural","confidence":5,"justification":"‹how would I send› asks for concrete method. Not_challenge_understanding because GET requests are generalizable; not_code_generation because no explicit code request."}}

**concept_guidance_non_procedural:**
[Learner] why do I need to add a delay [/Learner]
{{"result":"concept_guidance_non_procedural","confidence":5,"justification":"‹why do I need› asks for reasoning not procedure. Not_procedural because why not how; not_confirmation because open question not validation."}}

**challenge_understanding:**
[Learner] how do I get the color data for the challenge im on [/Learner]
{{"result":"challenge_understanding","confidence":4,"justification":"‹color data for the challenge› asks about challenge-specific artifact. Not_concept_guidance because color data is specific to this challenge; not_request_direction because asks about specific element."}}

**social_turn:**
[Learner] thank you [/Learner]
{{"result":"social_turn","confidence":5,"justification":"‹thank you› is social acknowledgment. Not_observations because no technical content; not_vague_request because no request."}}

**non_sequitur:**
[Learner] flag [/Learner]
{{"result":"non_sequitur","confidence":4,"justification":"‹flag› single ambiguous word with no clear intent. Not_social_turn because not greeting/thanks; not_vague_request because not a request pattern."}}

[Learner] lets analyze [/Learner]
{{"result":"non_sequitur","confidence":3,"justification":"‹lets analyze› fragmentary statement with unclear referent. Not_request_direction because no question asked; not_observations because no factual content reported."}}

---

### SELF-CHECK BEFORE RESPONDING ###

Before outputting JSON, verify:
1. Did I check the decision order and stop at the FIRST match?
2. Does my chosen code fit the heuristics (✓) and avoid exclusions (✘)?
3. Is my justification 20-100 words with ‹phrase› from marked message?
4. Did I explicitly contrast with at least one rejected alternative using "not_"?
5. Did I ONLY use content from the [*Learner*] marked message?
6. For "doesn't work" variants: Is it SENSAI attribution (→observations) or own problem (→debugging)?
7. For "is X wrong?": Is it yes/no question (→confirmation) or open diagnosis (→debugging)?

---

### INSTRUCTIONS ###

• Inspect the last Learner segment marked: [*Learner*] … [/*Learner*]
• **This is the ONLY message you are coding**
• Think step by step about learner intent
• Follow decision order strictly
• Output JSON only

### CRITICAL: MESSAGE ISOLATION ###
**The phrase in ‹ › must originate from the [*Learner*] marked message ONLY**

### CHALLENGE CONTEXT ###
The learner is currently working on the following challenge:

<challenge_description>
{challenge_description}
</challenge_description>

Use this context to:
- Distinguish challenge_understanding (challenge-specific) from concept_guidance (generalizable)
- Understand what tools/techniques are relevant to the current task
- Recognize challenge-specific artifacts, URLs, ports, or file references

Do NOT let the challenge description influence your classification beyond these purposes.

### SNIPPET ###
{snippet_string}
"""


# ==========================
# UTILITIES
# ==========================

def load_json(path: str) -> dict:
    with open(path, 'r') as f:
        return _normalize_raw(json.load(f))


def normalize_text(text: str) -> str:
    """normalize text for fuzzy comparison"""
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text


# pass 1 of 2: runs at load time on every raw learner message; pass 2
# (detect_description_chunk_pass) re-runs at chunk time on this pass's output
# with different heuristics. both passes shaped the frozen production runs --
# do not merge or reorder them.
def detect_description_load_pass(
    message: str,
    challenge_description: str,
    threshold: float = 0.95
) -> tuple[bool, bool, str]:
    """fuzzy detect pasted challenge descriptions, returns:
    (is_only_description, contains_description, cleaned_message)"""
    if not challenge_description or not message:
        return False, False, message

    norm_desc = normalize_text(challenge_description)
    norm_msg = normalize_text(message)

    # too-short descriptions match unreliably
    if len(norm_desc) < 20:
        return False, False, message

    similarity = SequenceMatcher(None, norm_msg, norm_desc).ratio()
    if similarity >= threshold:
        return True, True, "<pasted challenge description>"

    desc_len = len(norm_desc)
    msg_len = len(norm_msg)

    if desc_len > msg_len:
        return False, False, message

    best_match_start = -1
    best_match_end = -1
    best_similarity = 0

    window_sizes = [desc_len, int(desc_len * 1.1), int(desc_len * 0.9)]

    for window_size in window_sizes:
        if window_size > msg_len:
            continue
        for start in range(0, msg_len - window_size + 1, max(1, window_size // 10)):
            end = start + window_size
            window = norm_msg[start:end]
            sim = SequenceMatcher(None, window, norm_desc).ratio()
            if sim > best_similarity and sim >= threshold:
                best_similarity = sim
                best_match_start = start
                best_match_end = end

    if best_match_start >= 0:
        # map positions back to the original message (approximate)
        before_match = norm_msg[:best_match_start].strip()
        after_match = norm_msg[best_match_end:].strip()

        has_content_before = len(before_match) > 5
        has_content_after = len(after_match) > 5

        if not has_content_before and not has_content_after:
            return True, True, "<pasted challenge description>"

        cleaned_parts = []

        if has_content_before:
            ratio = best_match_start / len(norm_msg)
            approx_pos = int(len(message) * ratio)
            cleaned_parts.append(message[:approx_pos].strip())

        cleaned_parts.append("<pasted challenge description>")

        if has_content_after:
            ratio = best_match_end / len(norm_msg)
            approx_pos = int(len(message) * ratio)
            cleaned_parts.append(message[approx_pos:].strip())

        cleaned_message = " ".join(p for p in cleaned_parts if p)
        return False, True, cleaned_message

    return False, False, message


def load_json_and_convert(path: str) -> Optional[dict]:
    """load SENSAI json export into {dojo_id, module_id, challenge_id,
    challenge_description, interactions}"""
    data = load_json(path)

    if "tutor" not in data:
        return None

    if data["tutor"].get("gpt_failures"):
        return None

    challenge_description = data["tutor"].get("challenge_description", "")
    out = {
        "dojo_id": data["tutor"].get("dojo_id", ""),
        "module_id": data["tutor"].get("module_id", ""),
        "challenge_id": data["tutor"].get("challenge_id", ""),
        "challenge_description": challenge_description
    }

    ilist = sorted(
        (x for x in data["tutor"]["interactions"]
         if x["type"] != "FileBreak"
         and not (x["type"] == "TutorInteraction" and "thoughts" not in x)),
        key=lambda x: x["timestamp"]
    )

    # validate and fix alternation
    is_valid = True
    true_ilist = []

    for item in ilist:
        expected_type = ["LearnerInteraction", "TutorInteraction"][len(true_ilist) % 2]

        if item["type"] != expected_type:
            is_valid = False
            if item["type"] == "TutorInteraction":
                true_ilist.append({
                    "type": "LearnerInteraction",
                    "message": "[No message recorded]",
                    "terminal": "",
                    "file": "",
                    "timestamp": item['timestamp']
                })
                true_ilist.append(item)
            else:
                if true_ilist:
                    true_ilist.pop()
                true_ilist.append(item)
        else:
            true_ilist.append(item)

    final_list = ilist if is_valid else true_ilist

    # group into {learner_msg, tutor_msg, terminal, file} pairs
    interactions = []
    for i in range(0, len(final_list), 2):
        learner = final_list[i] if i < len(final_list) else {}
        tutor = final_list[i + 1] if i + 1 < len(final_list) else {}

        learner_msg = learner.get("message", "")

        is_only_desc, contains_desc, cleaned_msg = detect_description_load_pass(
            learner_msg, challenge_description
        )

        if is_only_desc:
            interactions.append({
                "learner_msg": "<pasted challenge description>",
                "tutor_msg": tutor.get("message", ""),
                "terminal": learner.get("terminal", ""),
                "file": learner.get("file", ""),
            })
        elif contains_desc:
            interactions.append({
                "learner_msg": cleaned_msg,
                "tutor_msg": tutor.get("message", ""),
                "terminal": learner.get("terminal", ""),
                "file": learner.get("file", ""),
            })
        else:
            interactions.append({
                "learner_msg": learner_msg,
                "tutor_msg": tutor.get("message", ""),
                "terminal": learner.get("terminal", ""),
                "file": learner.get("file", ""),
            })

    out["interactions"] = interactions

    return out


def truncate_string(string: str, max_len: int = 400, sep: str = "[...]") -> str:
    """truncate long strings"""
    if len(string) <= max_len:
        return string
    half_len = (max_len - len(sep)) // 2
    return string[:half_len] + sep + string[-half_len:]


def fuzzy_match_ratio(text1: str, text2: str) -> float:
    """get similarity ratio between two texts"""
    norm1 = normalize_text(text1)
    norm2 = normalize_text(text2)
    if not norm1 or not norm2:
        return 0.0
    return SequenceMatcher(None, norm1, norm2).ratio()


# pass 2 of 2: chunk-time detector; see the load-pass note above
def detect_description_chunk_pass(
    message: str,
    challenge_description: str,
    threshold: float = 0.95,
) -> tuple[bool, bool, str]:
    """detect pasted challenge descriptions, returns
    (is_only_description, contains_description, processed_message)."""
    if not challenge_description or not challenge_description.strip():
        return False, False, message

    message = message.strip()
    challenge_description = challenge_description.strip()

    overall_ratio = fuzzy_match_ratio(message, challenge_description)
    if overall_ratio >= threshold:
        return True, True, "<pasted challenge description>"

    desc_len = len(challenge_description)
    msg_len = len(message)

    if desc_len < 50 or msg_len < desc_len:
        # too short to detect reliably
        return False, False, message

    norm_desc = normalize_text(challenge_description)
    norm_msg = normalize_text(message)

    best_start = -1
    best_end = -1
    best_ratio = 0.0

    window_size = len(norm_desc)
    step = max(1, window_size // 10)

    for start in range(0, max(1, len(norm_msg) - window_size + 1), step):
        end = start + window_size
        window = norm_msg[start:end]
        ratio = SequenceMatcher(None, window, norm_desc).ratio()

        if ratio > best_ratio:
            best_ratio = ratio
            best_start = start
            best_end = end

    if best_ratio >= threshold and best_start >= 0:
        # map back to original positions (approximate)
        orig_start = int(best_start * len(message) / len(norm_msg))
        orig_end = int(best_end * len(message) / len(norm_msg))

        # expand to word boundaries
        while orig_start > 0 and message[orig_start - 1] not in ' \n\t':
            orig_start -= 1
        while orig_end < len(message) and message[orig_end] not in ' \n\t':
            orig_end += 1

        before = message[:orig_start].strip()
        after = message[orig_end:].strip()
        remaining = (before + " " + after).strip()

        # trivial remainder -> description-only
        remaining_nonwhitespace = ''.join(remaining.split())
        if len(remaining_nonwhitespace) < 12:
            return True, True, "<pasted challenge description>"

        processed = before + " <pasted challenge description> " + after
        processed = ' '.join(processed.split())

        without_placeholder = processed.replace("<pasted challenge description>", "")
        remaining_content = ''.join(without_placeholder.split())
        if len(remaining_content) < 12:
            return True, True, "<pasted challenge description>"

        return False, True, processed

    return False, False, message


# filter pasted challenge descriptions into req direction
PREFILTER_CODE = "request_direction"


# ==========================
# CHUNKING
# ==========================

class Chunk(pydantic.BaseModel):
    """conversation chunk centered on a learner message"""
    chunk_id: int  # index of the learner message (0, 2, 4, ...)
    messages: List[dict]  # interaction messages (learner + tutor)
    prev_messages: List[dict] = []  # previous interaction messages (for context)
    terminal: str = ""  # learner's terminal output
    file: str = ""  # learner's file content
    # challenge description handling
    should_prefilter: bool = False  # true if message is only challenge description
    original_learner_msg: str = ""
    processed_learner_msg: str = ""

    def to_string(self, do_truncate: bool = True, context_max_len: int = 500) -> str:
        """format the chunk for the prompt"""
        elems = []

        for message in self.prev_messages:
            content = str(message["content"]).strip()
            if do_truncate:
                content = truncate_string(content, max_len=400)
            elems.append(
                f'[{message["role"].upper()}] {content} [/{message["role"].upper()}]'
            )

        for i, message in enumerate(self.messages):
            content = str(message["content"]).strip()
            if do_truncate:
                content = truncate_string(content, max_len=400)

            is_target = (i == 0 and message["role"] == "Learner")
            marker = "*" if is_target else ""

            elems.append(
                f'[{marker}{message["role"].upper()}{marker}] {content} [/{marker}{message["role"].upper()}{marker}]'
            )

        result = "\n".join(elems)

        context_parts = []

        if self.terminal and self.terminal.strip():
            terminal_content = self.terminal.strip()
            if do_truncate:
                terminal_content = truncate_string(terminal_content, max_len=context_max_len)
            context_parts.append(f"[TERMINAL]\n{terminal_content}\n[/TERMINAL]")

        if self.file and self.file.strip():
            file_content = self.file.strip()
            if do_truncate:
                file_content = truncate_string(file_content, max_len=context_max_len)
            context_parts.append(f"[FILE]\n{file_content}\n[/FILE]")

        if context_parts:
            result += "\n\n### LEARNER CONTEXT ###\n" + "\n\n".join(context_parts)

        return result


def create_chunks(interactions: List[dict]) -> List[Chunk]:
    """one chunk per learner message: the current pair + the previous pair as
    context"""
    chunks = []

    for i, interaction in enumerate(interactions):
        chunk_id = i * 2

        messages = [
            {"role": "Learner", "content": interaction["learner_msg"]},
            {"role": "SENSAI", "content": interaction["tutor_msg"]}
        ]

        prev_messages = []
        if i > 0:
            prev_interaction = interactions[i - 1]
            prev_messages = [
                {"role": "Learner", "content": prev_interaction["learner_msg"]},
                {"role": "SENSAI", "content": prev_interaction["tutor_msg"]}
            ]

        chunks.append(Chunk(
            chunk_id=chunk_id,
            messages=messages,
            prev_messages=prev_messages,
            terminal=interaction.get("terminal", ""),
            file=interaction.get("file", ""),
        ))

    return chunks


def process_chunk_challenge_description(
    chunk: Chunk,
    challenge_description: str,
    threshold: float = 0.95,
) -> Chunk:
    """detect/replace a pasted challenge description in the chunk's learner
    message"""
    learner_msg = chunk.messages[0]["content"]
    chunk.original_learner_msg = learner_msg

    is_only_desc, contains_desc, processed_msg = detect_description_chunk_pass(
        learner_msg, challenge_description, threshold
    )

    chunk.should_prefilter = is_only_desc
    chunk.processed_learner_msg = processed_msg

    if contains_desc:
        chunk.messages[0]["content"] = processed_msg

    return chunk


# ==========================
# PROMPT BUILDING
# ==========================

def build_prompt(chunk: Chunk, challenge_description: str) -> str:
    """build classification prompt for a chunk"""
    return CLASSIFIER_PROMPT.format(
        challenge_description=challenge_description,
        snippet_string=chunk.to_string()
    )


def get_structured_response_format(response_model: Type[pydantic.BaseModel]) -> dict:
    """structured-output format for openai"""
    schema = response_model.model_json_schema()
    schema_name = schema.get("title", "ResponseFormat") or "ResponseFormat"
    schema["additionalProperties"] = False

    if "$defs" in schema:
        for def_name, def_schema in schema["$defs"].items():
            if "enum" in def_schema:
                def_schema["additionalProperties"] = False

    return {
        "type": "json_schema",
        "name": schema_name,
        "strict": True,
        "schema": schema,
    }


# ==========================
# INPUT NORMALIZATION
# ==========================

def _normalize_raw(raw):
    if isinstance(raw, dict) and "sensai" in raw and "tutor" not in raw:
        raw = dict(raw); raw["tutor"] = raw.pop("sensai")
    t = raw.get("tutor") if isinstance(raw, dict) else None
    if isinstance(t, dict):
        for it in t.get("interactions", []):
            if it.get("type") == "SensaiInteraction":
                it["type"] = "TutorInteraction"
    return raw


# ==========================
# MODEL APIS
# ==========================

SCHEMA = get_structured_response_format(ResponseFormat)
SCHEMA_NAME = SCHEMA["name"]
SCHEMA_JSON = SCHEMA["schema"]


def make_classifier(provider: str):
    """return classify_one(prompt) -> code for the chosen provider"""
    if provider == "openai":
        key = os.environ.get("OPENAI_API_KEY") or sys.exit("set OPENAI_API_KEY")
        client = OpenAI(api_key=key)
        params = {"reasoning": {"effort": REASONING_EFFORT}, "seed": SEED, "temperature": TEMPERATURE}
        text = {"format": SCHEMA, "verbosity": VERBOSITY}
        reported = []

        def classify_one(prompt):
            while True:
                try:
                    r = client.responses.create(model=MODEL_OPENAI, input=prompt, text=text,
                                                max_output_tokens=MAX_OUTPUT_TOKENS, store=False, **params)
                    break
                except (BadRequestError, TypeError) as e:
                    m = str(e).lower()
                    if "verbosity" in m and "verbosity" in text:
                        text.pop("verbosity"); print("  [dropped unsupported: verbosity]", flush=True)
                    elif (k := next((k for k in ("seed", "temperature", "reasoning") if k in m and k in params), None)):
                        params.pop(k); print(f"  [dropped unsupported: {k}]", flush=True)
                    else:
                        raise
            if not reported:
                reported.append(1)
                print(f"  [reproducibility params in effect: {sorted(params)}"
                      f"{' + verbosity' if 'verbosity' in text else ''}]", flush=True)
            return json.loads(r.output_text)["result"]

        return classify_one

    if provider == "openrouter":
        key = os.environ.get("OPENROUTER_API_KEY") or sys.exit("set OPENROUTER_API_KEY")
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)
        response_format = {"type": "json_schema",
                           "json_schema": {"name": SCHEMA_NAME, "strict": True, "schema": SCHEMA_JSON}}
        extra_body = {"provider": {"order": ["openai"], "allow_fallbacks": False}}
        params = {"seed": SEED, "temperature": TEMPERATURE, "reasoning_effort": REASONING_EFFORT}
        reported = []

        def classify_one(prompt):
            while True:
                try:
                    r = client.chat.completions.create(
                        model=MODEL_OPENROUTER, messages=[{"role": "user", "content": prompt}],
                        response_format=response_format, max_completion_tokens=MAX_OUTPUT_TOKENS,
                        extra_body=extra_body, **params)
                    break
                except (BadRequestError, TypeError) as e:
                    m = str(e).lower()
                    if (k := next((k for k in ("seed", "temperature", "reasoning_effort") if k in m and k in params), None)):
                        params.pop(k); print(f"  [dropped unsupported: {k}]", flush=True)
                    else:
                        raise
            if not reported:
                reported.append(1)
                print(f"  [reproducibility params in effect: {sorted(params)} + provider-pinned(openai)]", flush=True)
            return json.loads(r.choices[0].message.content)["result"]

        return classify_one

    sys.exit(f"unknown PROVIDER {provider!r} (use 'openai' or 'openrouter')")


# ==========================
# CLASSIFICATION
# ==========================

def input_json_paths(input_dir: str) -> List[str]:
    """all session JSONs under input_dir (recursive)"""
    return sorted(glob.glob(os.path.join(input_dir, "**", "*.json"), recursive=True))


def build_worklist(input_dir: str) -> List[dict]:
    items = []
    for path in input_json_paths(input_dir):
        sid = os.path.splitext(os.path.basename(path))[0]
        data = load_json_and_convert(path)
        if data is None:
            continue
        desc = data["challenge_description"]
        for chunk in create_chunks(data["interactions"]):
            process_chunk_challenge_description(chunk, desc)
            placeholder = chunk.processed_learner_msg.strip() == "<pasted challenge description>"
            if chunk.should_prefilter or placeholder:
                items.append({"session_id": sid, "chunk_id": chunk.chunk_id,
                              "prompt": None, "prefilter_code": PREFILTER_CODE})
            else:
                items.append({"session_id": sid, "chunk_id": chunk.chunk_id,
                              "prompt": build_prompt(chunk, desc), "prefilter_code": None})
    return items


def run_classification(classify_one, input_dir: str, out_path: str):
    items = build_worklist(input_dir)
    done = set()
    fresh = not os.path.exists(out_path) or os.path.getsize(out_path) == 0
    if not fresh:
        with open(out_path) as f:
            done = {(r["session_id"], str(r["chunk_id"])) for r in csv.DictReader(f)}
    todo = [it for it in items if (it["session_id"], str(it["chunk_id"])) not in done]
    fh = open(out_path, "a", newline=""); w = csv.writer(fh)
    if fresh:
        w.writerow(["session_id", "chunk_id", "result"]); fh.flush()
    print(f"{len(todo)} queries to classify (of {len(items)}; {len(done)} already done) -> {out_path}",
          flush=True)
    for i, it in enumerate(todo, 1):
        code = it["prefilter_code"] if it["prefilter_code"] else classify_one(it["prompt"])
        w.writerow((it["session_id"], it["chunk_id"], code)); fh.flush()
        if i % 25 == 0 or i == len(todo):
            print(f"  {i}/{len(todo)}", flush=True)
    fh.close()
    print("classification done ->", out_path, flush=True)


# ==========================
# EXPORT TO STAGE 02
# ==========================

# dojo sub-module id -> course module number (1-9)
MODULE_NUM = {
    "piping": 1, "permissions": 1, "users": 1, "path": 1, "variables": 1,
    "commands": 1, "globbing": 1, "chaining": 1, "processes": 1, "man": 1,
    "paths": 1, "hello": 1,
    "data-dealings": 2, "access-control": 2,
    "sql-playground": 3, "talking-web": 3,
    "web-security": 4,
    "assembly-crash-course": 5, "memory": 5, "debugging-refresher": 5,
    "your-first-program": 5, "building-a-web-server": 5, "introspecting": 5,
    "intercepting-communication": 6,
    "cryptography": 7,
    "reverse-engineering": 8,
    "binary-exploitation": 9,
}


def assemble_inputfeatures(codes_csv: str, input_dir: str, out_csv: str,
                           solves_csv: Optional[str] = None):
    # session metadata from the raw JSONs, keyed by filename stem (= session_id)
    meta = {}
    for path in input_json_paths(input_dir):
        d = load_json(path)
        s = d.get("tutor") or {}   # load_json normalizes the "sensai" key to "tutor"
        sid = os.path.splitext(os.path.basename(path))[0]
        meta[sid] = {"user_id": s.get("user_id", ""),
                     "module_num": MODULE_NUM.get(s.get("module_id", ""), "")}

    solved = {}
    if solves_csv:
        with open(solves_csv) as f:
            for r in csv.DictReader(f):
                solved[r["session_id"]] = r["solved"]

    rows, missing = [], set()
    with open(codes_csv) as f:
        for r in csv.DictReader(f):
            sid = r["session_id"]
            m = meta.get(sid)
            if m is None:
                missing.add(sid); continue
            rows.append([m["user_id"], sid, int(r["chunk_id"]) // 2 + 1, r["result"],
                         m["module_num"], solved.get(sid, "")])
    rows.sort(key=lambda x: (x[1], x[2]))

    with open(out_csv, "w", newline="") as g:
        w = csv.writer(g)
        w.writerow(["user_id", "session_id", "messagenum", "task_code", "module_num", "solved"])
        w.writerows(rows)

    print(f"wrote {out_csv}: {len(rows)} rows, {len(set(r[1] for r in rows))} sessions")
    if missing:
        print(f"  WARNING: {len(missing)} session(s) in classifier output had no JSON metadata")
    if not solves_csv:
        print("  NOTE: 'solved' left blank — join from the platform solves log before running 02")


# ==========================
# MAIN
# ==========================

def main():
    ap = argparse.ArgumentParser(
        description="Classify learner queries (stage 01) and assemble the stage-02 input.")
    ap.add_argument("--provider", default=PROVIDER, choices=["openai", "openrouter"])
    ap.add_argument("--input-dir", default=INPUT_DIR)
    ap.add_argument("--output-csv", default=OUTPUT_CSV)
    ap.add_argument("--inputfeatures-csv", default=STAGE2_INPUT_CSV)
    ap.add_argument("--solves-csv", default=SOLVES_CSV)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the prompt for the first query and exit (no API calls)")
    ap.add_argument("--assemble-only", action="store_true",
                    help="skip classification; assemble inputfeatures from an existing --output-csv (no API key needed)")
    a = ap.parse_args()

    if a.dry_run:
        items = build_worklist(a.input_dir)
        n_pre = sum(1 for it in items if it["prompt"] is None)
        print(f"# {len(items)} queries ({n_pre} prefiltered) from {a.input_dir!r}")
        first = next((it for it in items if it["prompt"] is not None), None)
        if first is None:
            print("# no non-prefiltered queries found")
        else:
            print(f"# prompt for first query (session {first['session_id']}, chunk {first['chunk_id']}):")
            print(first["prompt"])
        return

    if not a.assemble_only:
        classify_one = make_classifier(a.provider)
        run_classification(classify_one, a.input_dir, a.output_csv)
    assemble_inputfeatures(a.output_csv, a.input_dir, a.inputfeatures_csv, a.solves_csv)


if __name__ == "__main__":
    main()
