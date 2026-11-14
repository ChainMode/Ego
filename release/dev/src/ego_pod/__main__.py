"""``python -m ego_pod`` 入口，等价于 ``ego`` 命令。"""

import sys


def _main() -> int:
    # 延迟导入，避免在仅查询包元数据时拉起完整 CLI 依赖。
    from ego_pod.cli import main

    return main()


if __name__ == "__main__":
    sys.exit(_main())
