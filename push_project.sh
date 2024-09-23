#!/bin/bash

# Usage:
# ./push_project.sh [project_dir]

PROJECT_DIR="${1:-.}"
REPO_URL="https://github.com/yabtesfu/gan-lab-tensorflow.git"

if [ ! -d "$PROJECT_DIR" ]; then
  echo "Project directory not found: $PROJECT_DIR"
  exit 1
fi

cd "$PROJECT_DIR" || { echo "Failed to cd into $PROJECT_DIR"; exit 1; }

if [ ! -d ".git" ]; then
  git init
fi
git branch -M main

git config user.name "yabtesfu"
git config user.email "yabtesfu@gmail.com"

START_DATE="2024-09-23"
END_DATE="2025-01-11"

HOLIDAYS=(
  "2024-10-14"
  "2024-11-11"
  "2024-11-28"
  "2024-12-25"
  "2025-01-01"
)

WEEKEND_DAYS=(
  "2024-10-12"
  "2024-11-03"
  "2024-11-23"
  "2024-12-07"
  "2024-12-29"
  "2025-01-11"
)

MESSAGES=(
  "Add synthetic GAN dataset helpers"
  "Implement quadratic distribution sampler"
  "Build TensorFlow generator models"
  "Build discriminator feature head"
  "Add vanilla GAN losses"
  "Add WGAN gradient penalty"
  "Wire alternating training loop"
  "Add sample visualization helpers"
  "Track moment distance metric"
  "Add coverage metric"
  "Document GAN experiment plan"
  "Add CLI training entry point"
  "Refine training config validation"
  "Add reproducible data tests"
  "Prepare DCGAN model builders"
  "Add conditional GAN helper"
  "Tune replay buffer sampling"
  "Track MMD evaluation metric"
  "Add adaptive augmentation policy"
  "Wire TTUR schedule helpers"
  "Polish README workflow"
)

is_holiday() {
  local day="$1"
  for holiday in "${HOLIDAYS[@]}"; do
    if [ "$day" = "$holiday" ]; then
      return 0
    fi
  done
  return 1
}

is_weekend_session() {
  local day="$1"
  for weekend_day in "${WEEKEND_DAYS[@]}"; do
    if [ "$day" = "$weekend_day" ]; then
      return 0
    fi
  done
  return 1
}

big_day_commits() {
  case "$1" in
    "2024-09-23") echo $((RANDOM % 4 + 8)) ;;
    "2024-10-09") echo $((RANDOM % 4 + 10)) ;;
    "2024-10-28") echo $((RANDOM % 5 + 8)) ;;
    "2024-11-15") echo $((RANDOM % 4 + 9)) ;;
    "2024-12-03") echo $((RANDOM % 5 + 10)) ;;
    "2024-12-20") echo $((RANDOM % 4 + 8)) ;;
    "2025-01-07") echo $((RANDOM % 4 + 9)) ;;
    *) echo 0 ;;
  esac
}

next_day() {
  date -j -v+1d -f "%Y-%m-%d" "$1" "+%Y-%m-%d" 2>/dev/null ||
    date -d "$1 +1 day" "+%Y-%m-%d"
}

day_of_week() {
  date -j -f "%Y-%m-%d" "$1" "+%u" 2>/dev/null ||
    date -d "$1" "+%u"
}

CURRENT="$START_DATE"
END_NEXT=$(next_day "$END_DATE")

while [ "$CURRENT" != "$END_NEXT" ]; do
  DOW=$(day_of_week "$CURRENT")

  if is_holiday "$CURRENT"; then
    CURRENT=$(next_day "$CURRENT")
    continue
  fi

  if { [ "$DOW" = "6" ] || [ "$DOW" = "7" ]; } && ! is_weekend_session "$CURRENT"; then
    CURRENT=$(next_day "$CURRENT")
    continue
  fi

  BIG_DAY=$(big_day_commits "$CURRENT")
  if [ "$BIG_DAY" -gt 0 ]; then
    COMMITS_TODAY="$BIG_DAY"
  elif is_weekend_session "$CURRENT"; then
    COMMITS_TODAY=$((RANDOM % 2 + 1))
  else
    if [ $((RANDOM % 100)) -lt 28 ]; then
      CURRENT=$(next_day "$CURRENT")
      continue
    fi

    if [ $((RANDOM % 100)) -lt 5 ]; then
      COMMITS_TODAY=3
    elif [ $((RANDOM % 100)) -lt 30 ]; then
      COMMITS_TODAY=2
    else
      COMMITS_TODAY=1
    fi
  fi

  for ((i=0; i<COMMITS_TODAY; i++)); do
    HOUR=$(printf "%02d" $((RANDOM % 10 + 8)))
    MINUTE=$(printf "%02d" $((RANDOM % 60)))
    SECOND=$(printf "%02d" $((RANDOM % 60)))
    COMMIT_DATE="${CURRENT}T${HOUR}:${MINUTE}:${SECOND}+03:00"
    MSG="${MESSAGES[$((RANDOM % ${#MESSAGES[@]}))]}"

    echo "${COMMIT_DATE} - ${MSG}" >> history.txt

    git add .
    GIT_AUTHOR_DATE="$COMMIT_DATE" GIT_COMMITTER_DATE="$COMMIT_DATE" \
      git commit --allow-empty -m "$MSG"
  done

  CURRENT=$(next_day "$CURRENT")
done

if git remote get-url origin >/dev/null 2>&1; then
  git remote set-url origin "$REPO_URL"
else
  git remote add origin "$REPO_URL"
fi

git push -u origin main
