#!/bin/bash
# ARCHIVED -- -mandelbrot no longer exists in the active build (see
# ARCHIVE_NOTE_mandelbrot_jump.md in this directory). Running the
# "mandelbrot" half of this script now will fail with
# "error: unknown option -mandelbrot". Kept for reference: this is the
# script that produced the final, confound-free null result (t ~= 0.19).
# The -zone-force / registry-reset METHODOLOGY in here is still good and
# reusable for any future A/B test in this codebase -- only the specific
# -mandelbrot flag is gone.
# ============================================================
# compare_mandelbrot_zone.sh
#
# Zone-mode K-factor comparison: baseline jump selection vs.
# -mandelbrot (now wired to the REAL operational zone_id, not the
# old fixed zone_id=0 slice -- see mandelbrot_jump.h / RCKangaroo.cpp
# g_active_zone_id comments for what changed and why).
#
# Runs N independent solves for a SOLVED puzzle's known zone with
# baseline selection, then N with -mandelbrot, and reports the
# median K for each. Each run gets a completely fresh zone DP file
# so runs are statistically independent (otherwise DPs collected
# in run 1 would carry into run 2 and contaminate its K).
#
# Usage:
#   ./compare_mandelbrot_zone.sh [puzzle] [runs]
# Example (puzzle 80, 5 runs each):
#   ./compare_mandelbrot_zone.sh 80 5
# ============================================================

PUZZLE=${1:-80}
RUNS=${2:-5}
EXE="./rckangaroo"
LOG="compare_mandelbrot_p${PUZZLE}.csv"

GREEN="\033[1;32m"; YELLOW="\033[1;33m"; RED="\033[1;31m"; CYAN="\033[1;36m"; RESET="\033[0m"

# --- known-puzzle table (mirrors zone_mode.cpp's SOLVED_PUZZLES + your
#     solved_puzzles_commands_zones.txt) -- add rows here if you want to
#     spot-check a different puzzle. ---
case "$PUZZLE" in
  80)  PUBKEY="037e1238f7b1ce757df94faa9a2eb261bf0aeb9f84dbf81212104e78931c2a19dc"; START="80000000000000000000";           ZONE=82 ;;
  90)  PUBKEY="035c38bd9ae4b10e8a250857006f3cfd98ab15a6196d9f4dfd25bc7ecc77d788d5"; START="20000000000000000000000";      ZONE=40 ;;
  95)  PUBKEY="02967a5905d6f3b420959a02789f96ab4c3223a2c4d2762f817b7895c5bc88a045"; START="400000000000000000000000";    ZONE=28 ;;
  *) echo "Unknown puzzle $PUZZLE -- add its pubkey/start/zone to this script's case statement first."; exit 1 ;;
esac

ZONE_DP_FILE=$(printf "zone_dps_p%d_z%02d.kangs" "$PUZZLE" "$ZONE")

RANGE=$PUZZLE
ZONE_DURATION=${ZONE_DURATION:-5}   # minutes; override with ZONE_DURATION=NN env var

echo -e "${CYAN}================================================================${RESET}"
echo -e "${CYAN}  Mandelbrot zone-mode K comparison -- Puzzle $PUZZLE, zone $ZONE${RESET}"
echo -e "${CYAN}  $RUNS runs baseline, $RUNS runs -mandelbrot${RESET}"
echo -e "${CYAN}================================================================${RESET}"
echo ""
echo "run,mode,k_factor" > "$LOG"

run_one() {
    local mode_label=$1
    shift
    local extra_args=("$@")

    # zone_registry.bin persists dps_collected/sessions PER ZONE across every
    # process launch and drives round-robin zone selection for any zone
    # visit after the first in a run. Left in place, a run that didn't solve
    # on the first (known-zone) attempt could leave state that biases which
    # zone the NEXT run tries on retry -- a real confound between the
    # baseline and mandelbrot batches. Wipe it (and its text sibling) too,
    # not just the DP file, so every run starts from truly identical state.
    rm -f "$ZONE_DP_FILE" zone_registry.bin zone_registry.txt

    local output
    # -zone-force pins zone selection unconditionally for this whole process
    # run, bypassing GetKnownZone()/GetLowestDPZone() entirely -- belt and
    # suspenders on top of deleting zone_registry.bin above: even if
    # something else someday writes to that file mid-run, zone choice can't
    # be affected. Requires a rebuilt binary (RCKangaroo.cpp/zone_mode.cpp/
    # zone_registry.h) -- if you see "error: unknown option -zone-force",
    # rebuild first.
    output=$("$EXE" -pubkey "$PUBKEY" -range "$RANGE" -start "$START" \
                     -zone-mode -zone-duration "$ZONE_DURATION" -zone-force "$ZONE" \
                     -gpu 012 -cpu 64 "${extra_args[@]}" 2>&1 \
             | sed $'s/\x1b\\[[0-9;?]*[a-zA-Z]//g; s/\r//g')

    # Primary: the explicit solve line. Fallback: kfactor_log.csv last row.
    local k
    k=$(echo "$output" | grep -oP 'Point solved, K: \K[0-9]+\.[0-9]+' | head -1)
    if [ -z "$k" ]; then
        k=$(awk -F, 'NR>1 && NF>=3 && $3~/^[0-9]/{last=$3} END{print last}' kfactor_log.csv 2>/dev/null)
    fi

    if echo "$output" | grep -q "FATAL ERROR: SolvePoint found incorrect key"; then
        echo -e "${RED}WRONG KEY -- STOP, do not trust this build${RESET}"
        echo "$output" | tail -30
        exit 1
    fi

    if ! echo "$output" | grep -q "k\*G == pubkey  OK"; then
        echo -e "${YELLOW}warning: no 'k*G == pubkey OK' verification line found this run${RESET}"
    fi

    # For -mandelbrot runs: confirm the sort actually happened (i.e. the
    # binary was rebuilt with it). Without this, a stale binary would
    # silently give you the old magnitude-blind comparison again.
    if [ "$mode_label" = "mandelbrot" ] && ! echo "$output" | grep -q "EcJumps1 sorted by magnitude"; then
        echo -e "${RED}warning: 'EcJumps1 sorted by magnitude' line missing -- rebuild before trusting this run${RESET}"
    fi

    if [ -z "$k" ]; then
        echo -e "${RED}FAILED (no K parsed -- did it solve within zone_duration=$ZONE_DURATION min?)${RESET}"
        echo "  --- tail of output ---"
        echo "$output" | tail -15
        echo "$i,$mode_label,ERROR" >> "$LOG"
        return
    fi

    echo -e "K = ${GREEN}$k${RESET}"
    echo "$i,$mode_label,$k" >> "$LOG"
}

echo -e "${YELLOW}--- Baseline (no -mandelbrot) ---${RESET}"
for i in $(seq 1 "$RUNS"); do
    printf "Run %2d/%d ... " "$i" "$RUNS"
    run_one baseline
done

echo ""
echo -e "${YELLOW}--- Mandelbrot (zone-correct mapping) ---${RESET}"
for i in $(seq 1 "$RUNS"); do
    printf "Run %2d/%d ... " "$i" "$RUNS"
    run_one mandelbrot -mandelbrot
done

echo ""
echo -e "${CYAN}================================================================${RESET}"
echo -e "${CYAN}  Results${RESET}"
echo -e "${CYAN}================================================================${RESET}"

declare -A MEAN SD SE N
for mode in baseline mandelbrot; do
    vals=$(awk -F, -v m="$mode" '$2==m && $3!="ERROR" {print $3}' "$LOG")
    n=$(echo "$vals" | grep -c .)
    if [ "$n" -eq 0 ]; then
        echo -e "${RED}$mode: no successful runs${RESET}"
        continue
    fi
    # median (sort, take middle; average of two middles if even count)
    median=$(echo "$vals" | sort -n | awk -v n="$n" '{a[NR]=$1} END{
        if (n%2==1) printf "%.3f", a[(n+1)/2];
        else printf "%.3f", (a[n/2]+a[n/2+1])/2
    }')
    read -r mean sd se <<< "$(echo "$vals" | awk -v n="$n" '
        {s+=$1; a[NR]=$1}
        END{
            m=s/n
            for(i=1;i<=n;i++) ss += (a[i]-m)^2
            sd = (n>1) ? sqrt(ss/(n-1)) : 0
            se = sd/sqrt(n)
            printf "%.4f %.4f %.4f", m, sd, se
        }')"
    MEAN[$mode]=$mean; SD[$mode]=$sd; SE[$mode]=$se; N[$mode]=$n
    echo -e "${GREEN}$mode${RESET}: n=$n  median=$median  mean=$mean  sd=$sd  values=[$(echo $vals | tr '\n' ' ')]"
done

if [ -n "${MEAN[baseline]}" ] && [ -n "${MEAN[mandelbrot]}" ]; then
    echo ""
    awk -v mb="${MEAN[baseline]}" -v mm="${MEAN[mandelbrot]}" \
        -v sb="${SE[baseline]}" -v sm="${SE[mandelbrot]}" 'BEGIN{
        diff = mb - mm
        se_diff = sqrt(sb*sb + sm*sm)
        t = (se_diff > 0) ? diff/se_diff : 0
        printf "Difference of means: %.4f   combined SE: %.4f   t ~= %.3f\n", diff, se_diff, t
        if (t < 2.0)
            print "  -> NOT statistically significant at this sample size (need |t| >~ 2). Do not treat this as a confirmed effect yet."
        else
            print "  -> Plausibly significant -- but confirm with a larger N and a second puzzle/zone before trusting it."
    }'
fi

echo ""
echo "Raw data: $LOG"
echo -e "${CYAN}================================================================${RESET}"
