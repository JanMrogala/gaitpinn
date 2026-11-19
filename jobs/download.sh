#!/bin/bash
# Download Camargo et al. 2021 (Mendeley Data, CC BY 4.0, version 2) onto LUMI scratch.
# Runs on a LUMI login node in the background; check progress in $SCR/data/raw/download.log.
source "$(dirname "$0")/env.sh"
ssh "$LUMI_HOST" "mkdir -p '$SCR/data/raw' && cd '$SCR/data/raw' && nohup bash -s > download.log 2>&1 &" <<'REMOTE'
set -u
base=https://data.mendeley.com/public-files/datasets
while read -r ds id name size; do
  if [ -f "$name" ] && [ "$(stat -c %s "$name")" = "$size" ]; then echo "have $name"; continue; fi
  echo "get $name ($size bytes)"
  curl -fsSL --retry 5 -C - -o "$name" "$base/$ds/files/$id/file_downloaded"
  got=$(stat -c %s "$name")
  if [ "$got" = "$size" ]; then echo "ok $name"; else echo "SIZE MISMATCH $name: $got != $size"; fi
done <<LIST
fcgm3chfff 87d1e6e3-0f8e-4a59-b0a7-a22cc5cc2de4 README.txt 4970
fcgm3chfff d0f204d7-903c-4064-aff6-7080b2b5acb8 SubjectInfo.mat 1281
fcgm3chfff ff0bc55f-57cd-469e-8ae3-6736628d9731 scripts.zip 213893531
jj3r5f9pnf 2800004c-be9c-40fa-b9bd-50ad3ec1b085 Subjects_Part3_AB25-AB30.zip 4658784970
k9kvm5tn3f 2b976163-2870-458c-816d-30e07f17ad90 Subjects_Part2_AB15-AB24.zip 9508948781
fcgm3chfff 9c30f2e4-2d39-4d95-b691-1163c927455e Subjects_Part1_AB06-AB14.zip 10076406262
LIST
echo "DONE"
REMOTE
echo "download started; log: $SCR/data/raw/download.log"
