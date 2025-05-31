#!/usr/bin/env bash
# fix-math.sh — convert any lone dollar math into double dollar math for Kramdown MathJax
#               add proper blank lines around double dollar math blocks in Markdown
#               and replace common math symbols with their LaTeX command equivalents

set -euo pipefail

# parse options without using special characters in comments
DO_BACKUP=0
while (( "$#" )); do
  case "$1" in
    -b|--backup)
      DO_BACKUP=1
      shift
      ;;
    -*)
      echo "Unknown option: $1"
      echo "Usage: $0 [-b|--backup] file-or-dir [file-or-dir …]"
      exit 1
      ;;
    *)
      break
      ;;
  esac
done

# if no targets provided, show usage and exit
if [ "$#" -eq 0 ]; then
  echo "Usage: $0 [-b|--backup] file-or-dir [file-or-dir …]"
  exit 1
fi

# gather all markdown files from arguments, recursing into directories
declare -a MD_FILES=()
for target in "$@"; do
  if [ -d "$target" ]; then
    # find all files ending in dot m d under directory
    while IFS= read -r -d $'\0' mdfile; do
      MD_FILES+=("$mdfile")
    done < <(find "$target" -type f -name '*.md' -print0)
  else
    MD_FILES+=("$target")
  fi
done

# process each markdown file
for md in "${MD_FILES[@]}"; do
  if [ ! -f "$md" ]; then
    echo "Skipping: $md (not a file)"
    continue
  fi

  # create a timestamped backup if requested
  if [ "$DO_BACKUP" -eq 1 ]; then
    ts=$(date +"%F-T%H-%M-%S")
    cp -a -- "$md" "$md.bak.$ts"
    echo "Backup created: $md.bak.$ts"
  fi

  # use perl for in place editing; unify stray single dollar to double dollar,
  # ensure blank lines around block math, and replace common math symbols
  perl -i -0777 -pe '
    # convert any single dollar math delimiters into double dollar delimiters
    s/(?<!\$)\$(?!\$)/\$\$/g;

    my $yaml_delim      = 0;
    my $in_code         = 0;
    my $in_math         = 0;
    my $need_post_blank = 0;
    my @out;

    for my $line ( split /\n/ ) {

      # preserve YAML front matter marked by three hyphens at start
      if ( $yaml_delim < 2 && $line =~ /^---\s*$/ ) {
        $yaml_delim++;
        push @out, $line;
        next;
      }
      if ( $yaml_delim < 2 ) {
        push @out, $line;
        next;
      }

      # skip over fenced code blocks marked by backtick backtick backtick
      if ( $line =~ /^```/ ) {
        $in_code ^= 1;
        push @out, $line;
        next;
      }
      if ( $in_code ) {
        push @out, $line;
        next;
      }

      # after closing double dollar block, ensure one blank line if not already blank
      if ( $need_post_blank ) {
        push @out, "" unless $line =~ /^\s*$/;
        $need_post_blank = 0;
      }

      # detect lines that are exactly two dollar signs with optional spaces
      if ( $line =~ /^\s*\$\$\s*$/ ) {
        if ( !$in_math ) {
          # opening double dollar — ensure blank line above if not already blank
          push @out, "" if @out && $out[-1] !~ /^\s*$/;
          push @out, $line;
          $in_math = 1;
        }
        else {
          # closing double dollar — add blank line below
          push @out, $line;
          $in_math         = 0;
          $need_post_blank = 1;
        }
        next;
      }

      # if currently inside a math block (between double dollar delimiters)
      if ( $in_math ) {
        # replace three dots with latex command dots
        $line =~ s/\.\.\./\\dots/g;
        # replace two pipe characters with latex command Vert
        $line =~ s/\|\|/\\Vert/g;
        # replace backslash pipe sequence with latex command Vert
        $line =~ s/\\\|/\\Vert/g;
        # replace single pipe character with latex command vert
        $line =~ s/\|/\\vert/g;
        # replace asterisk with latex command ast
        $line =~ s/\*/\\ast/g;
        # replace tilde with latex command sim
        $line =~ s/~/\\sim/g;
      }

      push @out, $line;
    }

    # reassemble lines, ensuring final newline
    $_ = join("\n", @out) . "\n";
  ' "$md"

  echo "Processed: $md"
done
