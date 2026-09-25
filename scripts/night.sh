#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
#  Ночная серия TNNPM: три стадии ишемии на мелкой сетке, несколько ударов.
#
#    сетка     : эл. 160×160 (h = 0.05 мм), мех. 40×40 на 8×8 мм
#    ишемия    : круг r = 2 мм в (5, 4) мм; стимул — полоса x < 1 мм
#    ритм      : 4 удара, BCL 500 мс (1, 501, 1001, 1501), t_end = 2000 мс
#    точки     : run_000 — 15 мин (K_o 9.4, ATP 4.0, KmATP 0.38)
#                run_001 — 10 мин (K_o 8.0, ATP 4.5, KmATP 0.35)
#                run_002 — норма  (K_o 5.4, ATP 6.8, KmATP 0.09) — контроль
#
#  Три точки считаются ОДНОВРЕМЕННО, по NP рангов на каждую (3·NP ядер).
#  Потом `cardiac_em sweep` пропускает готовые точки и пишет сводку
#  runs/night/sweep.json — её читают 03/04 как обычную серию.
#
#  Запуск (из корня проекта, в окружении fenicsx-env):
#      NP=10 nohup bash scripts/night.sh > runs/night.log 2>&1 &
#      tail -f runs/night.log                       # общий ход
#      tail -f runs/night/run_000.log               # одна точка
#      python -m cardiac_em status runs/night/run_000
#
#  Повторный запуск пересчитывает только незавершённые точки.
# ═══════════════════════════════════════════════════════════════════════
# Всё тело — в функции main: bash разбирает её целиком до запуска.
# Иначе фоновые `run_point … &` при завершении сдвигают общую позицию
# чтения файла скрипта, и после `wait` bash дочитывает его не с того
# места («syntax error: unexpected end of file»).
main() {
set -uo pipefail
cd "$(dirname "$0")/.."
NP=${NP:-10}
PY=${PY:-python}
CFG=examples/tnnpm_night.json
SWEEP=examples/sweep_tnnpm_night.json
OUT=runs/night
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
df -h . | tail -1 | awk '{print "[диск] свободно: " $4 " (нужно ~2 ГБ)"}'

# ── проба: та же сетка, 20 мс подготовки + 20 мс счёта ─────────────────
echo "[$(ts)] проба"
t0=$(date +%s)
if ! mpirun "${MPI_OPTS[@]}" -n "$NP" "$PY" -m cardiac_em run "$CFG" \
        --out runs/night_smoke --t-end 20 --set preload.cell_relax_ms=20 \
        --overwrite --every 5; then
    echo "[$(ts)] ПРОБА НЕ ПРОШЛА — серия не запускается"; exit 1
fi
dt=$(( $(date +%s) - t0 ))
# 40 мс модели за dt с (включая сборку форм); точка — 2300 мс
echo "[оценка] проба ${dt} с → точка (и вся серия параллельно) ≲ $(( dt * 2300 / 40 / 60 )) мин"

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
        --out "$OUT/$name" --overwrite --every 50 \
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
}

main "$@"
exit $?
