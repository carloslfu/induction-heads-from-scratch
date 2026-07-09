#!/bin/sh
# Fetch a ~37MB public-domain novel corpus from Project Gutenberg for the
# word-level text runs (data/gutenberg.txt, not tracked in git — rerun this
# script to regenerate it). TinyShakespeare alone is too small at word
# granularity: a 688K-param model memorizes it (~300 epochs) and held-out
# loss climbs above chance. ~7.5M word-tokens brings that down to ~30
# epochs. Headers/footers are stripped via the standard *** START/END
# markers.
set -e
cd "$(dirname "$0")"

# id:title (title is a comment; the id is what's fetched)
BOOKS="2600 135 1184 145 2701 766 1023 1399 28054 996 1400 1342 1260 768
345 98 730 2554 599 2638 158 74 76 120 174"

rm -f gutenberg.txt
for id in $BOOKS; do
  url="https://www.gutenberg.org/cache/epub/$id/pg$id.txt"
  if curl -fsSL "$url" -o "pg$id.tmp"; then
    # keep only the text between the START and END markers
    awk '/\*\*\* START OF/{flag=1; next} /\*\*\* END OF/{flag=0} flag' \
      "pg$id.tmp" >> gutenberg.txt
    echo "ok  $id  ($(wc -c < pg$id.tmp | tr -d ' ') bytes)"
  else
    echo "FAILED $id ($url)" >&2
  fi
  rm -f "pg$id.tmp"
done
echo "total: $(wc -c < gutenberg.txt | tr -d ' ') bytes in gutenberg.txt"
