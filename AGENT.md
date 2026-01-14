# Agent Workflow for GPU Projects

## 1. Instance Setup
- Read `instance.txt` in the root directory to determine which GPU instance to use.  
- Connect to the instance for training.

## 2. Project Requirements
- Read `project_description.txt` carefully.  
- Follow **all requirements** and **test your work**.

## 3. Model Selection
- Default: GPT-2 may be insufficient.
- Use higher-quality model if needed (set HF_TOKEN environment variable)


## 4. File Management
- Keep a **local copy** of the project.  
- Use `rsync` or `scp` to **copy the project to the GPU instance**.  
- Train on the instance.  
- Copy results back to the local machine.  
- Commit and push to GitHub **from the local repo only**.

