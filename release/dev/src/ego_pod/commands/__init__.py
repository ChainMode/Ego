"""``ego`` 子命令实现包。

本包按子命令拆分 CLI 的实际业务入口，由 :mod:`ego_pod.cli` 在分发阶段
按需（惰性）导入：

* :mod:`ego_pod.commands.init` —— ``ego init`` 初始化工作目录（任务 4.3）。
* :mod:`ego_pod.commands.train` —— ``ego train`` 端到端训练编排（任务 11.1）。
* :mod:`ego_pod.commands.run` —— ``ego run`` 加载产出并进入交互对话（任务 10.3）。

各子命令入口统一接收 argparse 解析出的命名空间并返回 ``int`` 退出码，便于
CLI 顶层统一处理退出码与异常映射。
"""

__all__: list[str] = []
