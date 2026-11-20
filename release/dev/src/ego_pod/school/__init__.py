"""`ego_pod.school` —— `Ego School` 冷启动对话导入与伴生增量蒸馏模块。"""

from __future__ import annotations

from ego_pod.school.evolver import (
    DEFAULT_LEDGER_FILENAME,
    DEFAULT_MIN_CONSISTENCY,
    EvolutionResult,
    evolve_workspace,
)
from ego_pod.school.importer import (
    IMPORTED_RECORDS_FILENAME,
    LEDGER_CURSOR_FILENAME,
    SCHOOL_DIRNAME,
    SUPPORTED_IMPORT_FORMATS,
    SchoolImportResult,
    import_school_data,
    load_imported_school_records,
)

__all__ = [
    "SUPPORTED_IMPORT_FORMATS",
    "SCHOOL_DIRNAME",
    "IMPORTED_RECORDS_FILENAME",
    "LEDGER_CURSOR_FILENAME",
    "DEFAULT_LEDGER_FILENAME",
    "DEFAULT_MIN_CONSISTENCY",
    "SchoolImportResult",
    "EvolutionResult",
    "import_school_data",
    "load_imported_school_records",
    "evolve_workspace",
]
