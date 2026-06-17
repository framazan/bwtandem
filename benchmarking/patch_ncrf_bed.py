import sys

def patch_file():
    with open('scripts/utils/convert_to_bed.py', 'r') as f:
        content = f.read()
    
    # Update exclusion list in main() to also skip .bed files for ncrf to avoid overwriting our new beds
    old_exclude = "if f.endswith('.log') or f.endswith('.sh') or f.endswith('.settings'):"
    new_exclude = "if f.endswith('.log') or f.endswith('.sh') or f.endswith('.settings') or (tool == 'ncrf' and f.endswith('.bed')):"
    if old_exclude in content:
        content = content.replace(old_exclude, new_exclude)
    else:
        print("Could not find exclusion list!")
        sys.exit(1)

    with open('scripts/utils/convert_to_bed.py', 'w') as f:
        f.write(content)
    print("Patched successfully!")

if __name__ == '__main__':
    patch_file()
