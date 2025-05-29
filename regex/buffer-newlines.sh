#!/usr/bin/env bash
# fix-math.sh — convert any lone $…$ into $$…$$ for Kramdown/MathJax
#               and ensure proper blank lines around $$…$$ blocks in Markdown

set -euo pipefail

# parse options
DO_BACKUP=0
while (( "$#" )); do
  case "$1" in
    -b|--backup)
      DO_BACKUP=1
      shift
      ;;
    -*)
      echo "Unknown option: $1"
      echo "Usage: $0 [-b|--backup] file1.md [file2.md …]"
      exit 1
      ;;
    *)
      break
      ;;
  esac
done

# print usage if no files
if [ "$#" -eq 0 ]; then
  echo "Usage: $0 [-b|--backup] file1.md [file2.md …]"
  exit 1
fi

for md in "$@"; do
  if [ ! -f "$md" ]; then
    echo "Skipping: $md is not a file"
    continue
  fi

  # optional backup
  if [ "$DO_BACKUP" -eq 1 ]; then
    ts=$(date +"%F-T%H-%M-%S")
    cp -a -- "$md" "$md.bak.$ts"
    echo "Backup created: $md.bak.$ts"
  fi

  # in-place fix: first normalize stray single-$ to $$…$$, then ensure blank lines around block math
  perl -i -0777 -pe '
    # convert any single-dollar math delimiters into double-dollar
    s/(?<!\$)\$(?!\$)/\$\$/g;

    my $yaml_delim      = 0;
    my $in_code         = 0;
    my $in_math         = 0;
    my $need_post_blank = 0;
    my @out;

    for my $line ( split /\n/ ) {

      # preserve YAML front-matter
      if ( $yaml_delim < 2 && $line =~ /^---\s*$/ ) {
        $yaml_delim++;
        push @out, $line;
        next;
      }
      if ( $yaml_delim < 2 ) {
        push @out, $line;
        next;
      }

      # skip over fenced code blocks entirely
      if ( $line =~ /^```/ ) {
        $in_code ^= 1;
        push @out, $line;
        next;
      }
      if ( $in_code ) {
        push @out, $line;
        next;
      }

      # after closing $$ block, ensure one blank line
      if ( $need_post_blank ) {
        push @out, "" unless $line =~ /^\s*$/;
        $need_post_blank = 0;
      }

      # detect standalone $$ delimiters
      if ( $line =~ /^\s*\$\$\s*$/ ) {
        if ( !$in_math ) {
          # opening $$ → ensure blank above
          push @out, "" if @out && $out[-1] !~ /^\s*$/;
          push @out, $line;
          $in_math = 1;
        }
        else {
          # closing $$ → blank below
          push @out, $line;
          $in_math         = 0;
          $need_post_blank = 1;
        }
        next;
      }

      push @out, $line;
    }

    $_ = join("\n", @out) . "\n";
  ' "$md"

  echo "Processed: $md"
done
