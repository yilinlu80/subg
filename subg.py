cat << 'EOF' > ~/subg/subg
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ==============================================================================
# Script Name: subg V 1.5 (Gaussian 16 HPC Smart Submission Script)
# Author:      Yilin Lu
# Contact:     yilu@scripps.edu
#
# ------------------------------------------------------------------------------
# PREREQUISITES:
# - Python 3.10+ is recommended.
# - (Optional for AI features) Install the Gemini SDK: `pip install google-genai`
# - (Optional for AI features) Add your API key to ~/.bashrc or ~/.bash_profile:
#   export GOOGLE_API_KEY="your_api_key_here"
#
# INSTALLATION INSTRUCTIONS:
#
# Step 1: Clone the repository: `git clone https://github.com/yilinlu80/subg.git`
# Step 2: Add directory to PATH in ~/.bashrc: `export PATH="$HOME/subg/:$PATH"`
# Step 3: Grant execution permissions: `chmod +x subg.py`
# Step 4: Run script once to initialize the setup (e.g., `./subg.py`).
# Step 5: Verify installation: `subg -h`
# ------------------------------------------------------------------------------
# DESCRIPTION: 
#
# SUBG is a next-generation Gaussian 16 submission engine for Slurm clusters. 
# It minimizes human error and prevents wasted computational resources from 
# erroneous submissions by utilizing a hybrid validation method: FSM-based 
# strict formatting/physical rule verification paired with LLM-powered syntax correction.
# ==============================================================================

import os
import sys
import argparse
import re
import subprocess
import datetime

# ==========================================
# FEATURE TOGGLES & GLOBAL CONFIGS
# ==========================================
# MLOps Tracking: Enables audit trails for LLM prompts/responses to monitor AI behavior and hallucinations.
# Crucial for evaluating the accuracy of the stochastic LLM agent over time.
ENABLE_AI_LOG = True  

# Graceful Degradation Pattern: Ensures core HTC (High-Throughput Computing) submission 
# features remain functional in offline/air-gapped HPC environments without the Gemini SDK.
try:
    from google import genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

# Absolute paths for persistence caching and telemetry.
CONFIG_FILE = os.path.expanduser("~/.subg_config")
LOG_FILE = os.path.expanduser("~/subg/subg_history.log")

# Dynamic resolution ensures prompt templates are loaded relative to the script's execution context.
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
PROMPT_FILE_PATH = os.path.join(SCRIPT_DIR, "AutoCheckPrompt.txt")
CORRECT_PROMPT_FILE_PATH = os.path.join(SCRIPT_DIR, "AutoCorrectPrompt.txt")

# Domain Knowledge Base: O(1) lookup table for localized electron parity validation.
# Includes quantum chemistry pseudo-atoms (X, BQ, GH) often used for Basis Set 
# Superposition Error (BSSE) corrections, assigning them Z=0 to bypass electron counting heuristics.
ATOMIC_NUMBERS = {
    'H': 1, 'HE': 2, 'LI': 3, 'BE': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'NE': 10,
    'NA': 11, 'MG': 12, 'AL': 13, 'SI': 14, 'P': 15, 'S': 16, 'CL': 17, 'AR': 18,
    'K': 19, 'CA': 20, 'SC': 21, 'TI': 22, 'V': 23, 'CR': 24, 'MN': 25, 'FE': 26, 'CO': 27, 'NI': 28, 'CU': 29, 'ZN': 30,
    'GA': 31, 'GE': 32, 'AS': 33, 'SE': 34, 'BR': 35, 'KR': 36,
    'RB': 37, 'SR': 38, 'Y': 39, 'ZR': 40, 'NB': 41, 'MO': 42, 'TC': 43, 'RU': 44, 'RH': 45, 'PD': 46, 'AG': 47, 'CD': 48,
    'IN': 49, 'SN': 50, 'SB': 51, 'TE': 52, 'I': 53, 'XE': 54,
    'CS': 55, 'BA': 56, 'LA': 57, 'CE': 58, 'PR': 59, 'ND': 60, 'PM': 61, 'SM': 62, 'EU': 63, 'GD': 64, 'TB': 65, 'DY': 66,
    'HO': 67, 'ER': 68, 'TM': 69, 'YB': 70, 'LU': 71, 'HF': 72, 'TA': 73, 'W': 74, 'RE': 75, 'OS': 76, 'IR': 77, 'PT': 78,
    'AU': 79, 'HG': 80, 'TL': 81, 'PB': 82, 'BI': 83, 'PO': 84, 'AT': 85, 'RN': 86,
    'X': 0, 'BQ': 0, 'GH': 0 
}

# Inverse mapping for extracting standard element symbols from explicit Z-number formats (e.g., '6-0')
INV_ATOMIC_NUMBERS = {v: k for k, v in ATOMIC_NUMBERS.items() if k not in ['X', 'BQ', 'GH']}

def log_ai_conversation(prompt, response, log_type):
    """Audit logging for AI interactions. Crucial for debugging LLM hallucinations in scientific contexts."""
    if not ENABLE_AI_LOG:
        return
    try:
        with open("ai.log", "a", encoding="utf-8") as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"TIMESTAMP: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"TYPE: {log_type}\n")
            f.write(f"--- PROMPT ---\n{prompt}\n")
            f.write(f"--- RESPONSE ---\n{response}\n")
            f.write(f"{'='*60}\n")
    except Exception:
        pass

def get_autocheck_prompt():
    """Separation of concerns: Loads prompt templates externally to avoid hardcoding LLM instructions."""
    if os.path.exists(PROMPT_FILE_PATH):
        with open(PROMPT_FILE_PATH, "r", encoding="utf-8") as f:
            return f.read()
    elif os.path.exists("AutoCheckPrompt.txt"):
        with open("AutoCheckPrompt.txt", "r", encoding="utf-8") as f:
            return f.read()
        
    print("--> Error: 'AutoCheckPrompt.txt' not found!")
    print(f"--> Please ensure the file exists in your script directory: {SCRIPT_DIR}")
    sys.exit(1)

def get_autocorrect_prompt():
    """Loads auto-fix prompt template."""
    if os.path.exists(CORRECT_PROMPT_FILE_PATH):
        with open(CORRECT_PROMPT_FILE_PATH, "r", encoding="utf-8") as f:
            return f.read()
    elif os.path.exists("AutoCorrectPrompt.txt"):
        with open("AutoCorrectPrompt.txt", "r", encoding="utf-8") as f:
            return f.read()
            
    print("--> Error: 'AutoCorrectPrompt.txt' not found!")
    print(f"--> Please ensure the file exists in your script directory: {SCRIPT_DIR}")
    sys.exit(1)

def check_electronic_consistency(content):
    """
    Core Domain Heuristic: Strict Finite State Machine (FSM).
    Evaluates Gaussian input topologies in O(N) time complexity before invoking external LLM APIs.
    Validates physical constraints (electron parity) and topological constraints (connectivity/basis sets).
    From an AI4Science perspective, this acts as the "Deterministic Guardrail" to prevent 
    hallucinations or unnecessary token expenditures for trivial formatting errors.
    """
    # 1. Rule 1: EOF padding requirement. Fast O(1) string matching using regex.
    # Gaussian link modules strictly require at least two blank lines at EOF to trigger normal termination.
    if not re.search(r'\n[ \r\t]*\n[ \r\t]*\Z', content):
        return False, "Gaussian input formatting error: The file must end with at least two blank lines."

    # 2. Rule 2: Internal structural integrity. Prevents premature EOF parsing errors.
    body = content.rstrip()
    if re.search(r'\n[ \r\t]*\n[ \r\t]*\n', body):
        return False, "Gaussian input formatting error: Consecutive blank lines detected inside the file body. Sections must be separated by exactly ONE blank line."

    lines = content.splitlines()
    state = 'PRE_ROUTE' # FSM Initial State
    
    charge = None
    mult = None
    total_z = 0
    
    route_str = ""
    # Sets utilized for O(1) membership testing and efficient set-difference operations later
    elements_in_mol = set()
    elements_in_basis = set()

    # FSM Traversal (O(N) Time Complexity, O(1) Space Complexity beyond file I/O)
    # Efficiently parses multi-thousand atom coordinate blocks without loading heavy external dependencies.
    for i, line in enumerate(lines):
        l = line.strip()
        
        if state == 'PRE_ROUTE':
            if l.startswith('%'):
                continue # Link 0 commands bypass
            elif l.startswith('#'):
                state = 'IN_ROUTE'
                route_str += l + ' '
            elif l == '':
                return False, f"Line {i+1}: Illegal blank line detected before the route section (#)."
            else:
                return False, f"Line {i+1}: Invalid start. Missing or Wrong with Route (#) section."
                
        elif state == 'IN_ROUTE':
            if l == '':
                state = 'TITLE'
            else:
                route_str += l + ' '
                
        elif state == 'TITLE':
            if l == '':
                state = 'CHARGE_MULT'
                
        elif state == 'CHARGE_MULT':
            parts = l.split()
            if len(parts) >= 2:
                try:
                    charge = int(parts[0])
                    mult = int(parts[1])
                    state = 'ATOMS'
                except ValueError:
                    return False, f"Line {i+1}: Charge and Multiplicity must be integers immediately following the Title blank line."
            else:
                return False, f"Line {i+1}: Invalid Charge and Multiplicity definition. Expected two integers."
                
        elif state == 'ATOMS':
            if l == '':
                state = 'POST_ATOMS'
                continue
            
            parts = l.split()
            if len(parts) > 0:
                atom_str = parts[0]
                
                # Input sanitization against garbage encodings or typos
                if not re.match(r'^[A-Za-z0-9\-\.\(\)\*]+$', atom_str):
                    return False, f"Line {i+1}: Illegal or unrecognizable characters detected in atom definition '{atom_str}'."
                
                match = re.match(r'^([A-Za-z]{1,2})', atom_str)
                if match:
                    sym = match.group(1).upper()
                    if sym in ATOMIC_NUMBERS:
                        total_z += ATOMIC_NUMBERS[sym]
                        elements_in_mol.add(sym)
                    else:
                        return False, f"Line {i+1}: Unrecognized element symbol '{sym}'."
                else:
                    try:
                        z = int(atom_str.split('-')[0])
                        total_z += z
                        if z in INV_ATOMIC_NUMBERS:
                            elements_in_mol.add(INV_ATOMIC_NUMBERS[z])
                    except ValueError:
                        return False, f"Line {i+1}: Unrecognized atomic format '{atom_str}'."
                        
        elif state == 'POST_ATOMS':
            if l == '':
                # Keep scanning. Gaussian gen/genecp blocks or ECP blocks come after blank lines.
                continue
                
            parts = l.split()
            # Topology validation: Connectivity lists must contain an odd number of fields (1 Index + 2N Bonds)
            if len(parts) > 0 and parts[0].isdigit():
                if len(parts) % 2 == 0:
                    return False, f"Line {i+1}: Dangling bond detected in connectivity block ('{l}'). Missing a bond order or target atom."
            else:
                # Harvest elements defined in basis or ECP block for cross-referencing.
                for part in parts:
                    clean_part = part.upper()
                    if clean_part in ATOMIC_NUMBERS:
                        elements_in_basis.add(clean_part)
                    # Ignore negative elements (e.g. -F) representing exclusions
                    elif clean_part.startswith('-') and clean_part[1:] in ATOMIC_NUMBERS:
                        pass 

    # General Consistency Validations
    if charge is None or mult is None:
        return False, "Incomplete Gaussian input: Could not parse Charge and Multiplicity."

    # Physics Validation: Electron Count Parity
    # Formula: N_e = \sum(Z) - Charge. Spin topology dictates Multiplicity = 2S + 1.
    n_electrons = total_z - charge
    if mult == 0:
        return False, "Multiplicity cannot be 0."

    is_even_electrons = (n_electrons % 2 == 0)
    is_even_mult = (mult % 2 == 0)

    if is_even_electrons and is_even_mult:
        return False, f"Parity mismatch: Even number of electrons ({n_electrons}) requires an ODD multiplicity. Multiplicity {mult} specified."
    if not is_even_electrons and not is_even_mult:
        return False, f"Parity mismatch: Odd number of electrons ({n_electrons}) requires an EVEN multiplicity. Multiplicity {mult} specified."
    
    # Mathematical Set Validation for gen/genecp keywords
    route_lower = route_str.lower()
    requires_gen = bool(re.search(r'\b(gen|genecp)\b', route_lower))
    
    if requires_gen:
        # Set Difference (A - B): Finds atoms in the molecule that lack basis set mappings (O(1) average time)
        missing = {m for m in elements_in_mol if m not in ['X', 'BQ', 'GH']} - elements_in_basis
        if missing:
            return False, f"Gen/GenECP error: Elements {missing} are present in coordinates but missing from the basis set definition block."
        
        # Set Difference (B - A): Prevents fatal errors caused by over-defined basis blocks
        extra = elements_in_basis - elements_in_mol
        if extra:
            return False, f"Gen/GenECP error: Extra elements {extra} found in basis definition without a '-' prefix."

    return True, ""

def get_config():
    """Parses local configuration states."""
    config = {"email": None, "mailtype": "c"}
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            lines = f.read().strip().splitlines()
            for line in lines:
                if "=" in line:
                    k, v = line.split("=", 1)
                    config[k.strip()] = v.strip()
                elif "@" in line:
                    config["email"] = line.strip() 
    return config

def save_config(email, mailtype):
    """Persists operational settings across CLI sessions."""
    with open(CONFIG_FILE, 'w') as f:
        f.write(f"email={email if email else ''}\n")
        f.write(f"mailtype={mailtype}\n")

def log_submission(filename, job_id):
    """High-throughput computing (HTC) audit log for tracking successful Slurm dispatches."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", encoding="utf-8") as lf:
        lf.write(f"{timestamp} | JobID: {job_id} | {filename}\n")

def main():

    config = get_config()
    current_email = config["email"]
    current_mailtype = config["mailtype"]

    email_status = f"Saved Email: {current_email}" if current_email else "Saved Email: None (Set via -mail)"
    type_status = f"Default Type: {current_mailtype} (Set via -tmail)"

    border = "-" * 60
    description_text = (
        f"{border}\n"
        f"|   SUBG: GAUSSIAN 16 HPC SMART SUBMISSION SCRIPT V 1.5    |\n"
        f"{border}\n"
        f" {email_status} | {type_status}"
    )

    parser = argparse.ArgumentParser(
        description=description_text,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("filenames", nargs='*', help="Gaussian input files (e.g., mol1.gjf mol2.com)")
    parser.add_argument("-a", "--auto", action="store_true", help="Auto-submit .gjf or .com Gaussian input files")
    parser.add_argument("-m", "--mem", type=int, default=None, help="Memory in GB")
    parser.add_argument("-n", "--nproc", type=int, default=None, help="Number of CPU parallel processes")
    parser.add_argument("-t", "--time", type=int, default=72, help="Total calculation walltime in hours (default: 72)")
    parser.add_argument("-mail", "--email", type=str, default=None, help="Update to save email")
    parser.add_argument("-tmail", "--mailtype", type=str, default=current_mailtype, 
                        choices=['a', 's', 'e', 'f', 'c'], 
                        help="Email notification: (a)ll, (s)tart, (e)nd, (f)ail, (c)omplete/fail(default)")
    parser.add_argument("-l", "--log", type=str, choices=['on', 'off'], default='on', help="Submission logging (default: on)")
    
    # Validation toggles. Decouples physical checks (cheap) from AI checks (expensive).
    parser.add_argument("-fc", "--fsmcheck", type=str, choices=['on', 'off'], default='on', help="FSM check for input files (default: on)")
    parser.add_argument("-ac", "--aicheck", type=str, choices=['on', 'off'], default='off', help="AI Agent check for input files (default: off)")
    
    args = parser.parse_args()
    
    mailtype_provided = any(opt in sys.argv for opt in ["-tmail", "--mailtype"])
    config_updated = False

    if args.email:
        current_email = args.email
        config_updated = True
        print(f"--> Success: Email updated and saved as {current_email}")

    if mailtype_provided:
        current_mailtype = args.mailtype
        config_updated = True
        print(f"--> Success: Email notification trigger updated to '{current_mailtype}'")

    if config_updated:
        save_config(current_email, current_mailtype)
        if not args.filenames and not args.auto:
            sys.exit(0) 

    # Directory auto-scanner for automated high-throughput batching (HTC)
    if args.auto:
        pending_files = []
        for f in os.listdir('.'):
            if f.endswith('.gjf') or f.endswith('.com'):
                base_name = os.path.splitext(f)[0]
                if not (os.path.exists(base_name + '.log') or os.path.exists(base_name + '.out')):
                    pending_files.append(f)
        
        if not pending_files:
            print("--> No pending .gjf files found (all have corresponding .log or .out).")
            sys.exit(0)
            
        print("--> Found the following pending files:")
        for f in pending_files:
            print(f"    - {f}")
            
        choice = input(f"\nDo you want to submit these {len(pending_files)} jobs? (y/n, default 'y'): ").lower() or 'y'
        if choice != 'y':
            print("--> Auto-submission cancelled.")
            sys.exit(0)
        args.filenames = pending_files

    elif not args.filenames:
        if not config_updated:
            parser.print_help()
            print("\nError: You must specify filenames or use the -a/--auto flag.")
            sys.exit(1)
        else:
            sys.exit(0)

    mail_map = {'a': 'ALL', 's': 'BEGIN', 'e': 'END', 'f': 'FAIL', 'c': 'END,FAIL'}
    slurm_mail_type = mail_map.get(current_mailtype, 'ALL')

    for raw_filename in args.filenames:
        print(f"\n{raw_filename} submitting --------")
        
        base_name, ext = os.path.splitext(raw_filename)
        
        if ext:
            if ext.lower() not in ['.gjf', '.com']:
                print(f"--> Error: '{raw_filename}' is not a valid Gaussian input file (.gjf or .com). Skipping...")
                continue
            filename = raw_filename
            if not os.path.isfile(filename):
                print(f"--> Error: Target file '{filename}' not found. Skipping...")
                continue
        else:
            if os.path.isfile(f"{base_name}.gjf"):
                filename = f"{base_name}.gjf"
            elif os.path.isfile(f"{base_name}.com"):
                filename = f"{base_name}.com"
            else:
                print(f"--> Error: Target file '{base_name}.gjf' or '{base_name}.com' not found. Skipping...")
                continue
                
        job_name = os.path.splitext(filename)[0]

        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # ==========================================
        # Hybrid Validation method (FSM -> AI)
        # AI4Science Architecture: Cheap deterministic rules (FSM) act as guardrails 
        # to filter out obvious errors, reserving expensive stochastic LLM calls for complex debugging.
        # ==========================================
        api_key = os.environ.get("GOOGLE_API_KEY")
        
        # Local state overrides to allow runtime toggling without polluting global args
        current_fsmcheck = args.fsmcheck
        current_aicheck = args.aicheck
        
        validation_passed = False
        skip_file = False
        
        while not validation_passed:
            check_passed = True
            ai_result = ""
            
            # Phase 1: Zero-cost Python FSM strict format & parity check (Deterministic Heuristic)
            if current_fsmcheck == 'on':
                print("--> Initiating FSM Auto-check ...")
                is_consistent, py_err_msg = check_electronic_consistency(content)
                
                if not is_consistent:
                    print(f"--> FSM Auto-check: Failed!\n--> Reason: {py_err_msg}")
                    ai_result = py_err_msg # Passing structured FSM context to the LLM agent for auto-fixing
                    check_passed = False
                else:
                    print("--> FSM Auto-check: Passed. Structural and physical constraints verified.")
            
            # Phase 2: AI validation for complex structural/syntax errors (Stochastic Agent)
            # Acts as an expert system fallback for nuanced errors not caught by the FSM.
            if check_passed and current_aicheck == 'on':
                if not HAS_GENAI:
                    print("--> Warning: 'google-genai' not found. AI Auto-check skipped.")
                elif not api_key:
                    print("--> Warning: GOOGLE_API_KEY not set. AI Auto-check skipped.")
                else:
                    try:
                        client = genai.Client(api_key=api_key)
                        print("--> Initiating AI Auto-check Agent ...")
                        
                        prompt_text = get_autocheck_prompt()
                        full_prompt = f"{prompt_text}\n{content}"
                        
                        response = client.models.generate_content(
                            model='gemini-3.1-pro-preview',
                            contents=full_prompt
                        )
                        ai_result_check = response.text.strip()
                        
                        log_ai_conversation(full_prompt, ai_result_check, "AI_CHECK")
                        
                        if ai_result_check.lower().startswith("yes"):
                            print("--> AI Auto-check: Passed.")
                        else:
                            print(f"--> AI Auto-check: Failed!\n--> Reason: {ai_result_check}")
                            ai_result = ai_result_check
                            check_passed = False
                    except Exception as e:
                        print(f"--> AI Auto-check encountered an API error: {e}")
                        err_choice = input("Do you want to force submit anyway? (y/n): ").strip().lower()
                        if err_choice != 'y':
                            skip_file = True
                            break
                        else:
                            check_passed = True

            # Algorithm Engineer Note:
            # Short-Circuit Evaluation ensuring efficiency.
            # If FSM passes (check_passed = True) and AI Check is OFF (default behavior),
            # this immediately breaks the validation loop and proceeds to Slurm submission.
            # This guarantees deterministic O(N) execution without any unnecessary API overhead.
            if check_passed:
                validation_passed = True
                break

            # Phase 3: Resolution & Interactive Feedback Loop (Human-in-the-loop / Agentic Fix)
            print("--> Options: [M]anual edit, [A]uto fix, [F]orce submit, [Q]uit/Skip")
            chk_choice = input("Your choice (default Q): ").strip().upper()
            
            if chk_choice == 'F':
                print("--> Forcing submission...")
                validation_passed = True
                
            elif chk_choice == 'M':
                editor = os.environ.get('EDITOR', 'vi')
                print(f"--> Opening '{filename}' in {editor} for manual editing...")
                try:
                    subprocess.call([editor, filename])
                    with open(filename, 'r', encoding='utf-8') as f:
                        content = f.read()
                    print("--> Manual edit saved.")
                    
                    # Human-in-the-loop Decision Routing
                    print("--> Post-Edit Options: [1] FSM Check Only (default), [2] FSM + AI Check, [S]ubmit directly, [Q]uit/Skip")
                    post_edit_choice = input("Your choice (default 1): ").strip().upper()
                    
                    if post_edit_choice == 'S' or post_edit_choice == 'F':
                        print("--> Bypassing further validation. Proceeding to submission...")
                        validation_passed = True
                        break
                    elif post_edit_choice == 'Q':
                        print("--> Skipping submission.")
                        skip_file = True
                        break
                    elif post_edit_choice == '2':
                        print("--> Re-running with FSM + AI Check...")
                        current_fsmcheck = 'on'
                        current_aicheck = 'on'
                        continue 
                    else:
                        print("--> Re-running with FSM Check Only...")
                        current_fsmcheck = 'on'
                        current_aicheck = 'off'
                        continue 
                except Exception as e:
                    print(f"--> Error launching editor '{editor}': {e}")
                    print("--> Skipping submission.")
                    skip_file = True
                    break
                    
            elif chk_choice == 'A':
                print("--> Initiating AI Agent Auto-fix...")
                if not HAS_GENAI or not api_key:
                    print("--> Error: API key unavailable. Cannot Auto-fix.")
                    continue
                
                try:
                    client = genai.Client(api_key=api_key)
                    fix_prompt = get_autocorrect_prompt()
                    
                    # Context Injection: Fusing the FSM deterministic error with the raw file payload
                    full_fix_prompt = f"{fix_prompt}Failure Reason (FSM/AI Check):\n{ai_result}\nOriginal File Content:\n{content}"
                    
                    print("--> Calling AI Agent to correct the file...")
                    fix_response = client.models.generate_content(
                        model='gemini-3.1-pro-preview',
                        contents=full_fix_prompt
                    )
                    
                    raw_response = fix_response.text.strip()
                    log_ai_conversation(full_fix_prompt, raw_response, "AI_AUTO_FIX")
                    
                    # Markdown Sanitization: Extracts code block content from LLM response payload
                    if raw_response.startswith("```"):
                        lines = raw_response.splitlines()
                        if lines[-1].strip() == "```":
                            raw_response = "\n".join(lines[1:-1])
                        else:
                            raw_response = "\n".join(lines[1:])
                            
                    response_lines = raw_response.splitlines()
                    if len(response_lines) >= 2:
                        ai_modifications = response_lines[0]
                        new_content_fixed = "\n".join(response_lines[1:]).lstrip()
                    else:
                        ai_modifications = "AI modified the file (unable to parse modification summary line)."
                        new_content_fixed = raw_response
                    
                    print(f"--> AI Modifications:\n{ai_modifications}\n")
                    print(f"--> Fixed Gaussian Input:\n\n{new_content_fixed}")
                    
                    content = f"{new_content_fixed.rstrip()}\n\n\n"
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(content)
                        
                    print(f"--> Auto-fix applied and saved to '{filename}'.")
                    
                    # Interactive Routing Post-AI Intervention
                    print("--> Post-Fix Options: [1] FSM Check Only (default), [2] FSM + AI Check, [M]anual edit, [S]ubmit directly, [Q]uit/Skip")
                    post_fix_choice = input("Your choice (default 1): ").strip().upper()
                    
                    if post_fix_choice == 'M':
                        editor = os.environ.get('EDITOR', 'vi')
                        print(f"--> Opening '{filename}' in {editor} for manual editing...")
                        subprocess.call([editor, filename])
                        with open(filename, 'r', encoding='utf-8') as f:
                            content = f.read()
                        print("--> Manual edit saved. Re-running with FSM Check Only...")
                        current_fsmcheck = 'on'
                        current_aicheck = 'off'
                        continue 
                    elif post_fix_choice == 'S' or post_fix_choice == 'F':
                        print("--> Bypassing further validation. Proceeding to submission...")
                        validation_passed = True
                        break
                    elif post_fix_choice == 'Q':
                        print("--> Skipping submission.")
                        skip_file = True
                        break
                    elif post_fix_choice == '2':
                        print("--> Re-running with FSM + AI Check...")
                        current_fsmcheck = 'on'
                        current_aicheck = 'on'
                        continue
                    else:
                        print("--> Re-running with FSM Check Only...")
                        current_fsmcheck = 'on'
                        current_aicheck = 'off'
                        continue
                        
                except Exception as e:
                    print(f"--> AI Auto-fix encountered an error: {e}")
                    print("--> Skipping this submission...")
                    skip_file = True
                    break
            else:
                print("--> Skipping submission.")
                skip_file = True
                break

        if skip_file:
            continue
        # ==========================================

        # RegEx resource extractors tailored for Slurm node allocation mapping
        mem_match = re.search(rf'%(mem|-m)\s*=\s*(\d+)GB\b', content, re.IGNORECASE)
        nproc_match = re.search(rf'%(nprocshared|-p)\s*=\s*(\d+)\b', content, re.IGNORECASE)

        mem, nproc = args.mem, args.nproc

        if mem is None or nproc is None:
            if mem_match and nproc_match:
                file_mem, file_nproc = mem_match.group(2), nproc_match.group(2)
                print(f"--> Memory and Parallel settings in {filename}: %mem={file_mem}GB, %nproc={file_nproc}")
                choice = input(f"Proceed with {filename} memory and parallel settings? (y/n, default 'y'): ").lower() or 'y'
                
                if choice == 'y':
                    mem, nproc = int(file_mem), int(file_nproc)
                else:
                    mem, nproc = 24, 4
                    print(f"--> Applying default settings: %mem={mem}GB, %nproc={nproc}")
            else:
                mem, nproc = 24, 4
                print(f"--> No file-based settings found. Applying script defaults: {mem}GB, {nproc} cores.")
        
        mail_directives = ""
        if current_email:
            mail_directives = f"#SBATCH --mail-user={current_email}\n#SBATCH --mail-type={slurm_mail_type}"

        # Slurm Workload Manager Batch Script Compilation
        slurm_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes=1
#SBATCH --ntasks={nproc}
#SBATCH --time={args.time}:00:00
#SBATCH --mem={mem}GB
{mail_directives}

cd $SLURM_SUBMIT_DIR
module purge
module load gaussian/16

g16 {filename}
"""     

        slurm_filename = f"subg16_{job_name}.slurm"
        with open(slurm_filename, "w", encoding="utf-8") as f:
            f.write(slurm_content)
        
        # In-place regex substitution to prevent directive duplication in the input deck
        clean_content = re.sub(r'^\s*%(mem|-m)\b\s*=.*(\n|\Z)', '', content, flags=re.IGNORECASE | re.MULTILINE)
        clean_content = re.sub(r'^\s*%(nprocshared|-p|nproc)\b\s*=.*(\n|\Z)', '', clean_content, flags=re.IGNORECASE | re.MULTILINE)
        
        new_content = f"%mem={mem}GB\n%nprocshared={nproc}\n" + clean_content

        if new_content != content:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(new_content)

        # Non-blocking subprocess execution for job dispatching
        try:
            result = subprocess.run(["sbatch", slurm_filename], capture_output=True, text=True, check=True)
            output_str = result.stdout.strip()
            print(output_str) 
            
            job_id = "Unknown"
            id_match = re.search(r"Submitted batch job (\d+)", output_str)
            if id_match:
                job_id = id_match.group(1)

            print(f"--> '{job_name}'.gjf submitted successfully.")
            
            if args.log == 'on':
                log_submission(filename, job_id)

            # Garbage Collection: removes transient Slurm dispatcher files
            if os.path.exists(slurm_filename):
                os.remove(slurm_filename)
                
        except Exception as e:
            print(f"Submission failed for '{job_name}'.gjf: {e}")

if __name__ == "__main__":
    main()
EOF

echo "--> The SUBG V1.5 executable has been updated"
chmod +x ~/subg/subg
