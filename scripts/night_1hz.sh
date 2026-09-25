#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
#  Серия TNNPM на 1 Гц: три стадии ишемии, 10 ударов, установившийся ритм.
#
#    сетка     : эл. 160×160 (h = 0.05 мм), мех. 40×40 на 8×8 мм
#    ишемия    : круг r = 2 мм в (5, 4) мм; стимул — полоса x < 1 мм
#    ритм      : 10 ударов, BCL 1000 мс (1, 1001, …, 9001), t_end = 10 000 мс
#    вывод     : XDMF раз в 50 мс, снимки в 1-м и 10-м ударе, чекпоинт
#                раз в 250 мс; ~1.2 ГБ на точку
#    точки     : run_000 — 15 мин, run_001 — 10 мин, run_002 — норма
#
#  Сначала ЖДЁТ, пока закончится scripts/night.sh (первая серия), потом
#  считает три точки одновременно, по NP рангов на каждую.
#
#  Запуск (из корня проекта, в окружении fenicsx-env) — можно сразу,
#  пока идёт первая серия:
#      NP=8 nohup bash scripts/night_1hz.sh > runs/night_1hz.log 2>&1 &
#      tail -f runs/night_1hz.log
#      tail -2 runs/night_1hz/run_00*.log
#
#  Повторный запуск пересчитывает только незавершённые точки.
# ═══════════════════════════════════════════════════════════════════════
set -uo pipefail
cd "$(dirname "$0")/.."
NP=${NP:-10}
PY=${PY:-python}
CFG=examples/tnnpm_1hz.json
SWEEP=examples/sweep_tnnpm_1hz.json
OUT=runs/night_1hz
mkdir -p "$OUT"

# Несколько mpirun одновременно: Open MPI по умолчанию привязывает ранги
# к ядрам, начиная с нулевого, и три запуска сели бы на одни и те же
# ядра. Отключаем привязку; потоки BLAS/OpenMP — по одному на ранг.
NCPU=$(nproc)   # до export OMP_NUM_THREADS: GNU nproc его учитывает
MPI_OPTS=()
if mpirun --version 2>&1 | grep -qi "open mpi"; then
    MPI_OPTS=(--bind-to none)
fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

ts() { date '+%F %T'; }
echo "[$(ts)] ядер: $NCPU, рангов на точку: $NP, всего: $((3 * NP)), mpirun: ${MPI_OPTS[*]:-без опций}"
df -h . | tail -1 | awk '{print "[диск] свободно: " $4 " (нужно ~4 ГБ)"}'

# ── ждём первую серию (иначе делили бы с ней ядра) ─────────────────────
# Шаблон — только сам процесс `bash scripts/night.sh`, а не любая строка,
# где встречается это имя (tail, редактор, эта проверка).
FIRST='^bash (\S*/)?scripts/night\.sh( |$)'
if pgrep -f "$FIRST" > /dev/null; then
    echo "[$(ts)] жду окончания scripts/night.sh (pid $(pgrep -f "$FIRST" | tr '\n' ' '))…"
    while pgrep -f "$FIRST" > /dev/null; do sleep 60; done
    echo "[$(ts)] первая серия закончилась"
fi

# ── три точки параллельно ──────────────────────────────────────────────
run_point() {   # имя K_o ATP_i KmATP
    local name=$1
    if "$PY" - "$OUT/$name/run.json" <<'PYEOF' 2>/dev/null; then
import json, sys; sys.exit(0 if json.load(open(sys.argv[1])).get("status") == "finished" else 1)
PYEOF
        echo "[$(ts)] $name уже готова — пропуск"; return 0
    fi
    echo "[$(ts)] $name: K_o=$2 ATP_i=$3 KmATP=$4 → $OUT/$name.log"
    mpirun "${MPI_OPTS[@]}" -n "$NP" "$PY" -m cardiac_em run "$CFG" \
        --out "$OUT/$name" --overwrite --every 250 \
        --set "regions.0.overrides.cell:K_o=$2" \
        --set "regions.0.overrides.cell:ATP_i=$3" \
        --set "regions.0.overrides.cell:KmATP=$4" \
        > "$OUT/$name.log" 2>&1
    echo "[$(ts)] $name закончена, код $?"
}

run_point run_000 9.4 4.0 0.38 &
run_point run_001 8.0 4.5 0.35 &
run_point run_002 5.4 6.8 0.09 &
wait

# ── сводка серии (готовые точки не пересчитываются) ────────────────────
# Только если все точки готовы: иначе sweep стал бы досчитывать упавшую
# точку сам, на одном ранге.
if "$PY" - "$OUT" <<'PYEOF'; then
import json, sys
from pathlib import Path
ok = [json.loads((Path(sys.argv[1]) / n / "run.json").read_text()).get("status") == "finished"
      if (Path(sys.argv[1]) / n / "run.json").exists() else False
      for n in ("run_000", "run_001", "run_002")]
print("[точки]", dict(zip(("run_000", "run_001", "run_002"), ok)))
sys.exit(0 if all(ok) else 1)
PYEOF
    echo "[$(ts)] сводка серии"
    "$PY" -m cardiac_em sweep "$SWEEP" --quiet
    "$PY" -m cardiac_em status "$OUT"
    echo "[$(ts)] готово"
else
    echo "[$(ts)] не все точки готовы — смотрите $OUT/run_00*.log; повторный запуск скрипта досчитает остальные"
fi
