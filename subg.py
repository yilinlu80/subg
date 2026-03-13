cat << 'EOF' > ~/subg/subg
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ==============================================================================
# Script Name: subg (Gaussian 16 HPC Smart Submission Script)
# Author:      Yilin Lu
# Date:        03/12/2026
# Contact:     yilu@scripps.edu
#
# ------------------------------------------------------------------------------
# README & INSTALLATION INSTRUCTIONS:
#
# Step 1: Clone the subg.py script (git clone https://github.com/yilinlu80/subg.git)
#
# Step 2: Add that directory to your system's PATH so the command can be recognized globally.
#          (add in ~/.bashrc) Example: export PATH="$HOME/subg/:$PATH" 
#
# Step 3: Grant execution permissions to the script by running: chmod +x subg.py
#
# Step 4: Run the script once (e.g. './subg.py' or 'python3 subg.py' or 'subg.py' ) to 
#         initialize the setup. This will automatically generate the final 
#         'subg' executable file in the same directory.
#
# Step 5: Afterward, verify that the installation was successful and the 
#         script is functioning properly by running:
#         `subg -h`
# ------------------------------------------------------------------------------
#
# DESCRIPTION: 
# A smart command-line wrapper for submitting Gaussian 16 jobs to Slurm-based 
# HPC clusters. It features automatic resource extraction, batch submission, 
# persistent email configurations, and an auto-scan mode for pending calculations.
# ==============================================================================

import os
import sys
import argparse
import re
import subprocess

# Define the absolute path for storing the persistent email configuration
CONFIG_FILE = os.path.expanduser("~/.subg_config")

# Retrieve the saved email address from the configuration file if it exists
def get_saved_email():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            return f.read().strip()
    return None

# Save the provided email address to the configuration file for future use
def save_email(email):
    with open(CONFIG_FILE, 'w') as f:
        f.write(email)

def main():

    # Fetch the currently saved email to display its status in the help menu
    current_email = get_saved_email()
    email_status = f"Saved Email: {current_email}" if current_email else "Saved Email: None (Optional: Set via -mail)"

    # Define the visual layout and text for the boxed help menu description
    border = "-" * 60
    description_text = (
        f"{border}\n"
        f"|   SUBG: GAUSSIAN 16 HPC SMART SUBMISSION SCRIPT V0.2    |\n"
        f"{border}\n"
        f" {email_status}"
    )

    # Initialize the argument parser with the custom boxed description
    parser = argparse.ArgumentParser(
        description=description_text,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Configure all available command-line arguments and flags for the script
    parser.add_argument("filenames", nargs='*', help="Gaussian input files (e.g., mol1.gjf mol2.gjf)")
    parser.add_argument("-a", "--auto", action="store_true", help="Auto-submit .gjf files without .log or .out equivalents")
    parser.add_argument("-m", "--mem", type=int, default=None, help="Memory allocation in GB")
    parser.add_argument("-n", "--nproc", type=int, default=None, help="Number of CPU cores")
    parser.add_argument("-t", "--time", type=int, default=72, help="Walltime in hours (default: 72)")
    parser.add_argument("-mail", "--email", type=str, default=None, help="Update saved email")
    parser.add_argument("-tmail", "--mailtype", type=str, default="c", 
                        choices=['a', 's', 'e', 'f', 'c'], 
                        help="Notification trigger: (a)ll, (s)tart, (e)nd, (f)ail, (c)omplete/fail(default)")
    
    args = parser.parse_args()

    # Handle the auto-scan logic if the '-a' or '--auto' flag is triggered
    if args.auto:
        pending_files = []
        for f in os.listdir('.'):
            if f.endswith('.gjf'):
                base_name = f[:-4]

                # Add the file to the pending list if neither a .log nor an .out file exists for it
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

    # Exit the script and show the help menu if no filenames and no auto-flag are provided
    elif not args.filenames:
        parser.print_help()
        print("\nError: You must specify filenames or use the -a/--auto flag.")
        sys.exit(1)

    # Determine which email address to use based on user input or saved configuration
    email = None
    if args.email:
        save_email(args.email)
        email = args.email
        print(f"--> Success: Email updated and saved as {email}")
    elif current_email:
        email = current_email
    else:
        print("Note: No saved email address found.")
        print("You can save your email next time using: subg [file] -mail your_email@scripps.edu")

    # Map the single-character email triggers to the standard Slurm notification types
    mail_map = {'a': 'ALL', 's': 'BEGIN', 'e': 'END', 'f': 'FAIL', 'c': 'END,FAIL'}
    slurm_mail_type = mail_map.get(args.mailtype, 'ALL')

    # Loop through all identified files to process and submit them sequentially
    for raw_filename in args.filenames:
        filename = raw_filename
        print(f"\n{filename} submitting --------")
        
        # Ensure the file has the correct .gjf extension and exists in the directory
        if not filename.endswith('.gjf'):
            filename = f"{filename}.gjf"
        if not os.path.isfile(filename):
            print(f"Error: Target file {filename} not found. Skipping...")
            continue
            
        job_name = filename[:-4]

        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()

        # Scan the file content for existing memory and processor directives
        mem_match = re.search(rf'%(mem|-m)\s*=\s*(\d+)GB\b', content, re.IGNORECASE)
        nproc_match = re.search(rf'%(nprocshared|-p)\s*=\s*(\d+)\b', content, re.IGNORECASE)

        # Reset the memory and core variables for each iteration to prevent crossover from previous files
        mem, nproc = args.mem, args.nproc

        # Resolve conflicts between command-line arguments, file-based settings, and default script values
        if mem is None or nproc is None:
            if mem_match and nproc_match:
                file_mem, file_nproc = mem_match.group(2), nproc_match.group(2)
                print(f"--> Resource directives found: %mem={file_mem}GB, %nproc={file_nproc}")
                choice = input(f"Proceed with these settings for {filename}? (y/n, default 'y'): ").lower() or 'y'
                
                if choice == 'y':
                    mem, nproc = int(file_mem), int(file_nproc)
                else:
                    mem, nproc = 24, 4
                    print(f"--> Applying default settings: %mem={mem}GB, %nproc={nproc}")
            else:
                mem, nproc = 24, 4
                print(f"--> No file-based settings found. Applying script defaults: {mem}GB, {nproc} cores.")
        
        # Generate the specific Slurm email directives only if an email address is configured
        mail_directives = ""
        if email:
            mail_directives = f"#SBATCH --mail-user={email}\n#SBATCH --mail-type={slurm_mail_type}"

        slurm_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes=1
#SBATCH --ntasks={nproc}
#SBATCH --time={args.time}:00:00
#SBATCH --mem={mem}GB
{mail_directives}

input={job_name}
cd $SLURM_SUBMIT_DIR
module purge
module load gaussian/16

g16 ${{input}}.gjf
"""     

        # Define a unique Slurm script filename to prevent overwriting during batch submissions
        slurm_filename = f"subg16_{job_name}.slurm"
        with open(slurm_filename, "w", encoding="utf-8") as f:
            f.write(slurm_content)
        
        # Clean all historical memory and processor directives from the original file content
        clean_content = re.sub(r'^\s*%(mem|-m)\b\s*=.*(\n|\Z)', '', content, flags=re.IGNORECASE | re.MULTILINE)
        clean_content = re.sub(r'^\s*%(nprocshared|-p|nproc)\b\s*=.*(\n|\Z)', '', clean_content, flags=re.IGNORECASE | re.MULTILINE)
        
        # Prepend the newly resolved directives to the top of the cleaned content
        new_content = f"%mem={mem}GB\n%nprocshared={nproc}\n" + clean_content

        # Write the updated content back to the file only if changes were actually made
        if new_content != content:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(new_content)

        # Submit the generated Slurm script to the cluster's job scheduler
        try:
            subprocess.run(["sbatch", slurm_filename], check=True)
            print(f"--> '{job_name}'.gjf submitted successfully.")
            
            # Automatically clean up the temporary Slurm script if the submission is successful
            if os.path.exists(slurm_filename):
                os.remove(slurm_filename)
                
        except Exception as e:

            # Retain the Slurm script for debugging purposes if the submission fails
            print(f"Submission failed for '{job_name}'.gjf: {e}")

if __name__ == "__main__":
    main()
EOF

# Print a confirmation message indicating the update is complete
echo "--> The SUBG executable has been updated."

# Grant execution permissions to the newly generated script
chmod +x ~/subg/subg




