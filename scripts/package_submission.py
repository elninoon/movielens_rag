"""只打包明确列出的代码与说明，排除密钥及不可再分发的数据。"""
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'deliverables'


def main():
    OUTPUT.mkdir(exist_ok=True)
    target = OUTPUT / 'MovieLens_RAG_源码.zip'
    files = [ROOT / name for name in ['README.md', 'requirements.txt', '.env.example', '.gitignore', 'run.py']]
    for directory, suffixes in [('src', {'.py'}), ('tests', {'.py'}), ('scripts', {'.py', '.mjs'}), ('docs', {'.md'})]:
        files.extend(path for path in (ROOT / directory).rglob('*') if path.suffix in suffixes and '__pycache__' not in path.parts)
    files = [path for path in files if path.name != 'ai_notes.md']
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            archive.write(path, 'movielens_rag/' + path.relative_to(ROOT).as_posix())
    print(target)
    print(f'共 {len(files)} 个代码和说明文件，不含 .env、data、cache、outputs。')


if __name__ == '__main__':
    main()
