# subg: A Gaussian 16 HPC Smart Submission Script

A small, smart command-line wrapper designed for batch submitting Gaussian 16 jobs to Slurm-based HPC clusters. 

**Author:** Yilin Lu  
**Contact:** yilu@scripps.edu  

---

## ✨ Features

- **Integrated Workflow:** Automatically parses and cleans resource directives (`%mem`, `%nprocshared`) in your `.gjf` files, manages persistent Slurm email notifications, and auto-deletes temporary scripts to keep your workspace tidy.
- **Batch Submission:** Seamlessly submit multiple files at once.
- **Auto-Scan Submission (`-a`):** Automatically scans the current directory for pending `.gjf` files (those without corresponding `.log` or `.out` outputs) and queues them for bulk submission.

---

## 🚀 Installation & Setup

**Step 1: Download & Clone** 
Clone the subg.py script (git clone https://github.com/yilinlu80/subg.git)

**Step 2: Add to PATH** 
Add that directory to your system's PATH so the command can be recognized globally. (add in ~/.bashrc) Example: export PATH="$HOME/subg/:$PATH" 

**Step 3: Make Executable & Initialize** 
Grant execution permissions to the script: `chmod +x subg.py`. 
Then, run `./subg.py` (or `python3 subg.py` `subg.py`) to automatically generate the final `subg` executable command.

**Step 4: Verify Installation** 
Verify that everything is working properly by pulling up the help menu: `subg -h`



