import os
import re

def rebrand(directory):
    count = 0
    for root, dirs, files in os.walk(directory):
        # skip .git and node_modules and target and build
        dirs[:] = [d for d in dirs if d not in ['.git', 'node_modules', 'target', 'build', 'release_bins']]
        for file in files:
            # only process text files
            if file.endswith(('.md', '.html', '.ts', '.js', '.rs', '.py', '.ps1', '.css', '.c', '.h', '.cpp', '.hpp', '.toml', '.json')):
                path = os.path.join(root, file)
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    if 'WeView' in content or 'weview' in content:
                        new_content = content.replace('WeView', 'WeView').replace('weview', 'weview')
                        
                        with open(path, 'w', encoding='utf-8') as f:
                            f.write(new_content)
                        count += 1
                except Exception as e:
                    print(f"Failed to process {path}: {e}")
                    
    print(f"Rebranded {count} files.")

if __name__ == '__main__':
    rebrand('.')
