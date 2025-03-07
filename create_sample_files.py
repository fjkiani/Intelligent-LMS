import os

# Create data directory if it doesn't exist
os.makedirs("../data", exist_ok=True)

# Create a simple markdown file
with open("../data/devops-roadmap.md", "w") as f:
    f.write("""# DevOps Roadmap
    
## Popular Programming Languages for DevOps
- Python: Versatile language used for automation scripts and tools
- Go: Efficient for building containerized applications and microservices
- JavaScript/Node.js: Used for web applications and serverless functions
- Ruby: Popular for configuration management with tools like Chef
- Bash: Essential for shell scripting and automation

## Learning Resources
- Udemy courses on DevOps
- GitHub repositories with example projects
- Documentation for tools like Docker, Kubernetes, and Terraform
- Online communities like Stack Overflow and Reddit
""")

print("Sample files created. You'll still need to provide PDF and PPTX files.") 