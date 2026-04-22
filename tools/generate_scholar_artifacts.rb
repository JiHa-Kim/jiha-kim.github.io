#!/usr/bin/env ruby
# frozen_string_literal: true

require_relative "../_plugins/scholar_artifact_cache"

root = File.expand_path("..", __dir__)
path = Jekyll::ScholarArtifactCache.write_manifest!(root: root)
puts "Wrote #{path}"
