import os
from pathlib import Path

# Extensions to include
CODE_EXTENSIONS = {".py", ".pyx", ".pxd"}

def save_code_files_to_txt(root_dir, output_file="all_code_files.txt"):
    root_path = Path(root_dir)
    if not root_path.exists():
        print(f"Directory '{root_dir}' does not exist.")
        return

    with open(output_file, 'w', encoding='utf-8') as out_f:
        for file_path in root_path.rglob('*'):
            if file_path.is_file() and file_path.suffix in CODE_EXTENSIONS:
                relative_path = file_path.relative_to(root_path)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        out_f.write(f"\n--- {relative_path} ---\n")
                        out_f.write(content + "\n")
                except Exception as e:
                    out_f.write(f"\n--- Failed to read {relative_path}: {e} ---\n")

    print(f"All code files have been saved to '{output_file}'.")

if __name__ == "__main__":
    current_dir = Path.cwd()  # Current working directory
    save_code_files_to_txt(current_dir)
