#!/usr/bin/env bash
# fix-math.sh — convert lone dollar math to double dollars for Kramdown/MathJax
#               add blank lines around block math
#               and replace common math glyphs with LaTeX command equivalents
#               (always terminating each command with a space)

set -euo pipefail

# option parsing (no special chars in comments)
DO_BACKUP=0
while (( "$#" )); do
  case "$1" in
    -b|--backup) DO_BACKUP=1; shift ;;   # make timestamped .bak copy
    -*) echo "Unknown option: $1"; echo "Usage: $0 [-b] file-or-dir …"; exit 1 ;;
    *)  break ;;
  esac
done

if [ "$#" -eq 0 ]; then
  echo "Usage: $0 [-b] file-or-dir …"
  exit 1
fi

# gather *.md files recursively
declare -a MD_FILES=()
for target in "$@"; do
  if [ -d "$target" ]; then
    while IFS= read -r -d '' md; do MD_FILES+=("$md"); done < <(find "$target" -type f -name '*.md' -print0)
  else
    MD_FILES+=("$target")
  fi
done

for md in "${MD_FILES[@]}"; do
  [ -f "$md" ] || { echo "Skipping: $md (not a file)"; continue; }

  if [ "$DO_BACKUP" -eq 1 ]; then
    ts=$(date +'%F-T%H-%M-%S')
    cp -a -- "$md" "$md.bak.$ts"
    echo "Backup → $md.bak.$ts"
  fi

  perl -0777 -i -pe '
    ####################################################################
    # 1. Promote every solitary $ to $$ so Kramdown treats it as math.  #
    ####################################################################
    s/(?<!\$)\$(?!\$)/\$\$/g;

    ####################################################################
    # 2. Define a helper that rewrites the guts of any math expression #
    #    and *always* ends every inserted command with a space.        #
    ####################################################################
    sub fixmath {
      local $_ = shift;

      # order matters: longest / most specific first
      s/\.\.\./\\dots /g;          # … → \dots␠
      s/\\\|/\\Vert /g;            # \| → \Vert␠   (escaped already)
      s/\|\|/\\Vert /g;            # || → \Vert␠
      s/(?<!\\)\|/\\vert /g;       # |  → \vert␠   (skip \|)
      s/\*/\\ast /g;               # *  → \ast␠
      s/~/\\sim /g;                # ~  → \sim␠
      return $_;
    }

    ####################################################################
    # 3. Rewrite math in two situations                                #
    #    a) block math:   $$ … $$ on its own line(s)                   #
    #    b) inline math:  text $$ … $$ text                            #
    ####################################################################

    # a) process block math line-by-line while preserving blank lines
    my @out;
    my $yaml = 0;          # still inside YAML front matter?
    my $code = 0;          # inside ``` fenced code?
    my $inblock = 0;       # between stand-alone $$ lines?
    my $need_post = 0;     # need blank line after closing block?

    for my $line (split /\n/) {

      # YAML front matter passes through unchanged
      if ( $yaml < 2 && $line =~ /^---\s*$/ ) { $yaml++; push @out,$line; next }
      if ( $yaml < 2 )                         { push @out,$line; next }

      # fenced code passes through unchanged
      if ( $line =~ /^```/ ) { $code ^= 1; push @out,$line; next }
      if ( $code )           { push @out,$line; next }

      # insert blank line *after* closing $$ if needed
      if ( $need_post ) { push @out,"" unless $line =~ /^\s*$/; $need_post=0 }

      # detect stand-alone $$ delimiters
      if ( $line =~ /^\s*\$\$\s*$/ ) {
        if ( !$inblock ) {
          push @out,"" if @out && $out[-1] !~ /^\s*$/;   # blank line above
          push @out,$line; $inblock=1;
        } else {
          push @out,$line; $inblock=0; $need_post=1;
        }
        next;
      }

      if ( $inblock ) { $line = fixmath($line) }         # mutate block line
      else {
        # b) inline segments: $$ … $$ inside an ordinary line
        $line =~ s/\$\$(.*?)\$\$/"\$\$".fixmath($1)."\$\$"/ge;
      }

      push @out,$line;
    }

    $_ = join("\n",@out)."\n";
  ' "$md"

  echo "Processed: $md"
done
