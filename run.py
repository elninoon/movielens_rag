"""项目命令行入口。"""
import sys
from pathlib import Path

# Windows 代理证书沿用系统信任链，不关闭 HTTPS 验证。
try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).parent / 'src'))
from movie_rag.cli import main

if __name__ == '__main__':
    main()
