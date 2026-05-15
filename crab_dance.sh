#!/bin/bash

clear

ORANGE='\033[38;5;208m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
MAGENTA='\033[1;35m'
BOLD='\033[1m'
RESET='\033[0m'

FRAMES=(
'  /V\
(o . o)
 )   (
(__|__)'

' \(V)/
 (o.o)
  ) (
(__|__)'

'  /V\
(o . o)
(    )
(__|__)'

'-(V)(V)-
 (o_o)
  ) (
(__|__)'

'  /V\
(o . o)
 (    )
(__|__)'

'  /V\
(~ v ~)
 )   (
  \_/'
)

NOTES=('note' 'note2' 'note3' 'note4')
NOTE_CHARS=('J' 'JJ' 'j' '~J')

HEIGHT=$(tput lines)
WIDTH=$(tput cols)
STAGE_ROW=$(( HEIGHT - 7 ))
MAX_POS=$(( WIDTH - 22 ))
MIN_POS=4
POS=8
DIRECTION=1
FRAME_IDX=0
FRAME_COUNT=${#FRAMES[@]}
TICK=0
TITLE_IDX=0

declare -a N_ROW N_COL N_CHAR

add_note() {
  N_ROW+=( $(( STAGE_ROW - 5 )) )
  N_COL+=( $(( POS + RANDOM % 10 )) )
  local chars=('*' '+' 'o' '~')
  N_CHAR+=( "${chars[$(( RANDOM % 4 ))]}" )
}

update_notes() {
  local -a nr nc nch
  for ((i=0; i<${#N_ROW[@]}; i++)); do
    tput cup "${N_ROW[$i]}" "${N_COL[$i]}"
    printf '   '
    local r=$(( N_ROW[$i] - 1 ))
    if (( r > 2 )); then
      tput cup $r "${N_COL[$i]}"
      printf "${YELLOW}${N_CHAR[$i]}${RESET}"
      nr+=($r)
      nc+=("${N_COL[$i]}")
      nch+=("${N_CHAR[$i]}")
    fi
  done
  N_ROW=( "${nr[@]}" )
  N_COL=( "${nc[@]}" )
  N_CHAR=( "${nch[@]}" )
}

TITLE_COLORS=("$ORANGE" "$YELLOW" "$CYAN" "$MAGENTA")
TITLES=('*** CRAB RAVE ***' '~~~ CRAB BOOGIE ~~~' '<<< SHELL SHAKER >>>' '>>> PINCER PARTY <<<')

draw_title() {
  local msg="${TITLES[$(( TICK / 8 % 4 ))]}"
  local col=$(( (WIDTH - ${#msg}) / 2 ))
  tput cup 1 $col
  printf "${BOLD}${TITLE_COLORS[$TITLE_IDX]}%-40s${RESET}" "$msg"
  TITLE_IDX=$(( (TITLE_IDX + 1) % 4 ))
}

draw_crab() {
  local frame="$1"
  local col="$2"

  for ((r = STAGE_ROW - 4; r <= STAGE_ROW; r++)); do
    tput cup $r 0
    printf '%*s' "$WIDTH" ''
  done

  local row=$(( STAGE_ROW - 3 ))
  while IFS= read -r line; do
    tput cup $row $col
    printf "${ORANGE}${BOLD}%-22s${RESET}" "$line"
    (( row++ ))
  done <<< "$frame"

  # animated wavy floor
  tput cup $(( STAGE_ROW + 1 )) 0
  local floor=''
  for ((i=0; i<WIDTH; i++)); do
    case $(( (i + TICK) % 6 )) in
      0|5) floor+='~' ;;
      1|4) floor+='-' ;;
      2|3) floor+='~' ;;
    esac
  done
  printf "${CYAN}${floor}${RESET}"

  # press hint on first frames
  if (( TICK < 30 )); then
    local hint='  ctrl+c to exit  '
    tput cup $(( STAGE_ROW + 2 )) $(( (WIDTH - ${#hint}) / 2 ))
    printf "${MAGENTA}${hint}${RESET}"
  fi
}

tput civis
trap 'tput cnorm; tput clear; exit' INT TERM

while true; do
  draw_title
  update_notes
  (( TICK % 5 == 0 )) && add_note
  draw_crab "${FRAMES[$FRAME_IDX]}" "$POS"

  FRAME_IDX=$(( (FRAME_IDX + 1) % FRAME_COUNT ))
  POS=$(( POS + DIRECTION * 3 ))
  if (( POS >= MAX_POS )); then
    POS=$MAX_POS
    DIRECTION=-1
  elif (( POS <= MIN_POS )); then
    POS=$MIN_POS
    DIRECTION=1
  fi

  (( TICK++ ))
  sleep 0.15
done
