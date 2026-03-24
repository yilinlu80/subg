Here is the revised `README.md`. I've polished the phrasing to seamlessly integrate the "well-optimized subg AI Agent" terminology, ensuring it sounds professional, proprietary, and perfectly aligned with the product's advanced architecture.

-----

# 🚀 SUBG: Next-Gen Gaussian 16 HPC Smart Submission Engine

**SUBG (V1.5)** is a powerful, AI-enhanced command-line engine designed for batch submitting Gaussian 16 jobs to Slurm-based HPC clusters.

It minimizes human error and prevents wasted computational resources by utilizing a **Hybrid Validation Architecture**: combining a deterministic Finite State Machine (FSM) for strict physical/formatting rule verification, paired with an LLM-powered expert system for complex syntax correction.

**Author:** Yilin Lu  
**Contact:** yilu@scripps.edu

-----

## ✨ Key Features

  * 🛡️ **Hybrid FSM + AI Validation:** A zero-cost deterministic FSM instantly catches physical and formatting errors (e.g., charge/multiplicity parity, missing blank lines, `gen` block mappings). For nuanced structural errors, the well-optimized subg AI Agent serves as an expert fallback.
  * 🤖 **AI Auto-Correction (Human-in-the-Loop):** If errors are detected, SUBG pauses to offer interactive resolutions. You can manually edit, force-submit, or let the well-optimized subg AI Agent automatically analyze and rewrite the faulty input file.
  * 📂 **Smart Auto-Scan (`-a`):** Automatically detects pending `.gjf` or `.com` files in your directory (files missing corresponding `.log` or `.out` outputs) and queues them for bulk submission.
  * ⚙️ **Resource Auto-Parsing:** Automatically parses, cleans, and deduplicates resource directives (`%mem`, `%nprocshared`) in your input files.
  * 📬 **Persistent Slurm Configurations:** Saves your email and notification preferences globally so you only have to configure them once.
  * 📊 **Comprehensive Telemetry:** Tracks successful submissions with specific Slurm `JobID`s in a local history log, and maintains an MLOps audit trail (`ai.log`) to monitor AI behavior.

-----

## 🛠️ Prerequisites

  * **Environment:** Python 3.10+ is recommended.
  * **AI Features (Optional):** To enable the well-optimized subg AI Agent for auto-correction, install the required SDK:
    ```bash
    pip install google-genai
    ```
  * **API Key (Optional):** Add your Google API key to your `~/.bashrc` or `~/.bash_profile`:
    ```bash
    export GOOGLE_API_KEY="your_api_key_here"
    ```

-----

## 🚀 Installation

**Step 1: Clone the Repository**

```bash
git clone https://github.com/yilinlu80/subg.git
```

**Step 2: Add to PATH**
Add the directory to your system's PATH in your `~/.bashrc` so the command is recognized globally.

```bash
export PATH="$HOME/subg/:$PATH"
```

**Step 3: Grant Permissions & Initialize**
Grant execution permissions and run the script once to generate the final executable.

```bash
chmod +x ~/subg/subg.py
~/subg/subg.py
```

**Step 4: Verify Installation**
Verify the setup by pulling up the help menu.

```bash
subg -h
```

-----

## 💻 Usage & Examples

SUBG is designed to be highly granular. You can run basic submissions or leverage its full AI auditing capabilities.

### Common Workflows

1.  **Submit Specific Files:**
    ```bash
    subg molecule1.gjf complex2.com
    ```
2.  **Auto-Scan & Batch Submit (Finds all uncalculated files):**
    ```bash
    subg -a
    ```
3.  **Submit with Full AI & FSM Validation Enabled:**
    ```bash
    subg target.gjf -fc on -ac on
    ```
4.  **Update Persistent Email Configuration:**
    ```bash
    subg -mail myemail@example.edu -tmail c
    ```

### 🎛️ CLI Argument Reference

| Argument | Flag | Description | Default |
| :--- | :--- | :--- | :--- |
| **Files** | `filenames` | Target `.gjf` or `.com` files to submit. | None |
| **Auto-Scan** | `-a`, `--auto` | Auto-submit pending input files without output logs. | `False` |
| **Memory** | `-m`, `--mem` | Override memory allocation (in GB). | `24` |
| **Processors** | `-n`, `--nproc`| Override CPU core count. | `4` |
| **Walltime** | `-t`, `--time` | Total calculation walltime in hours. | `72` |
| **Email** | `-mail` | Set/Update your Slurm notification email address. | `None` |
| **Mail Type** | `-tmail` | Slurm trigger: `(a)ll`, `(s)tart`, `(e)nd`, `(f)ail`, `(c)omplete/fail`. | `c` |
| **FSM Check** | `-fc` | Toggle physical & strict format validation (`on`/`off`). | `on` |
| **AI Check** | `-ac` | Toggle LLM-powered syntax validation (`on`/`off`). | `off` |
| **Log** | `-l`, `--log` | Toggle JobID submission logging (`on`/`off`). | `on` |

-----

## 🧠 Architecture: The Hybrid Validation Engine

For those interested in the underlying data flow, SUBG V1.5 implements a multi-phase "AI4Science" validation pipeline:

1.  **Phase 1: Deterministic FSM (O(N) Time Complexity)**
      * Instantly parses multi-thousand-atom blocks without external dependencies.
      * Verifies strict physical guardrails: Electron count parity (`ΣZ - Charge`) vs. Spin Multiplicity.
      * Conducts Set-Difference operations to ensure `gen`/`genecp` basis blocks match atomic coordinates.
2.  **Phase 2: Stochastic LLM Agent (Expert Fallback)**
      * If FSM passes but `-ac` is enabled, the well-optimized subg AI Agent scans for nuanced logic/syntax errors.
3.  **Phase 3: Human-in-the-Loop Routing**
      * If an error triggers, users choose between launching a local `$EDITOR`, requesting an AI auto-fix via context-injected prompts, or forcing the job through.

-----

*Built to streamline High-Throughput Computational Chemistry.*
