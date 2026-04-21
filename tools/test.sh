#!/usr/bin/env bash
#
# Build and test the site content
#
# Requirement: html-proofer, jekyll
#
# Usage: See help information

set -eu

SITE_DIR="_site"

_config="_config.yml"

_baseurl=""
_fast="false"
_test_root=""
_test_config=""

cleanup() {
  if [[ -n "${_test_root:-}" && -d "$_test_root" ]]; then
    rm -rf "$_test_root"
  fi
}

prepare_test_root() {
  _test_root="$(mktemp -d)"
  trap cleanup EXIT

  git ls-files -co --exclude-standard -z \
    | rsync -a --from0 --files-from=- ./ "$_test_root/"

  if [[ $_config == *","* ]]; then
    IFS=","
    read -ra config_array <<<"$_config"
    local mapped_configs=()

    for config_path in "${config_array[@]}"; do
      mapped_configs+=("$_test_root/$config_path")
    done

    _test_config="$(IFS=,; echo "${mapped_configs[*]}")"
  else
    _test_config="$_test_root/$_config"
  fi
}

help() {
  echo "Build and test the site content"
  echo
  echo "Usage:"
  echo
  echo "   bash $0 [options]"
  echo
  echo "Options:"
  echo '     -c, --config   "<config_a[,config_b[...]]>"    Specify config file(s)'
  echo "     -f, --fast               Fast iterative build for local development (no minification, incremental, no htmlproofer)."
  echo "     -h, --help               Print this information."
}

read_baseurl() {
  if [[ $_config == *","* ]]; then
    # multiple config
    IFS=","
    read -ra config_array <<<"$_config"

    # reverse loop the config files
    for ((i = ${#config_array[@]} - 1; i >= 0; i--)); do
      _tmp_baseurl="$(grep '^baseurl:' "${config_array[i]}" | sed "s/.*: *//;s/['\"]//g;s/#.*//")"

      if [[ -n $_tmp_baseurl ]]; then
        _baseurl="$_tmp_baseurl"
        break
      fi
    done

  else
    # single config
    _baseurl="$(grep '^baseurl:' "$_config" | sed "s/.*: *//;s/['\"]//g;s/#.*//")"
  fi
}

main() {
  read_baseurl

  if [[ "$_fast" == "true" ]]; then
    # fast build
    JEKYLL_ENV=development bundle exec jekyll b \
      -d "$SITE_DIR$_baseurl" -c "$_config" --incremental
  else
    prepare_test_root

    # build from a clean copy of tracked + non-ignored files
    JEKYLL_ENV=production bundle exec jekyll b \
      -s "$_test_root" -d "$_test_root/$SITE_DIR$_baseurl" -c "$_test_config"

    # test
    bundle exec htmlproofer "$_test_root/$SITE_DIR" \
      --disable-external \
      --allow-hash-href \
      --ignore-empty-alt \
      --ignore-urls "/^http:\/\/127.0.0.1/,/^http:\/\/0.0.0.0/,/^http:\/\/localhost/"
  fi
}

while (($#)); do
  opt="$1"
  case $opt in
  -c | --config)
    _config="$2"
    shift
    shift
    ;;
  -f | --fast)
    _fast="true"
    shift
    ;;
  -h | --help)
    help
    exit 0
    ;;
  *)
    # unknown option
    help
    exit 1
    ;;
  esac
done

main
