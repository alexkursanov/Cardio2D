"""
Связь между электрической и механической задачами.
====================================================

Единственная забота слоя — перенести значение, посчитанное на одной
сетке, туда, где оно нужно на другой. Ни одна из сторон при этом не
должна знать об устройстве другой: солвер механики получает готовый
массив на своих ячейках и не подозревает, что он пришёл с более мелкой
сетки.

Состав:
    transfer.py — операторы переноса DG0 → DG0 и выбор стратегии
"""

from __future__ import annotations

from .transfer import (
    AveragingTransfer,
    FieldTransfer,
    NearestCellTransfer,
    make_transfer,
)

__all__ = [
    "FieldTransfer",
    "AveragingTransfer",
    "NearestCellTransfer",
    "make_transfer",
]
