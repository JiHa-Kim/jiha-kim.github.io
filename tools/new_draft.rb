#!/usr/bin/env ruby
# frozen_string_literal: true

require "fileutils"
require "optparse"

options = {
  categories: [],
  tags: [],
  drafts_dir: "collections/_drafts"
}

parser = OptionParser.new do |opts|
  opts.banner = "Usage: ruby tools/new_draft.rb TITLE [options]"

  opts.on("--slug SLUG", "Override the generated file slug") { |v| options[:slug] = v }
  opts.on("--description TEXT", "Set the draft description") { |v| options[:description] = v }
  opts.on("--category NAME", "Add a category; repeat up to two times") { |v| options[:categories] << v }
  opts.on("--tag NAME", "Add a tag; repeat as needed") { |v| options[:tags] << v }
  opts.on("--drafts-dir DIR", "Drafts directory, default: collections/_drafts") { |v| options[:drafts_dir] = v }
  opts.on("-h", "--help", "Print this help") do
    puts opts
    exit 0
  end
end

parser.parse!

title = ARGV.join(" ").strip

if title.empty?
  warn parser
  exit 1
end

def title_case(text)
  text.split(/\s+/).map { |word| word[0].upcase + word[1..].to_s }.join(" ")
end

def slugify(text)
  text.downcase
      .gsub(/['"]/, "")
      .gsub(/[^a-z0-9]+/, "-")
      .gsub(/\A-|-+\z/, "")
end

def yaml_list(items, fallback_count: 1)
  values = items.empty? ? Array.new(fallback_count, nil) : items
  values.map { |item| item ? "  - #{item}" : "  -" }.join("\n")
end

title = title_case(title)
slug = options[:slug] || slugify(title)

unless slug.match?(/\A[a-z0-9]+(?:-[a-z0-9]+)*\z/)
  warn "Invalid slug: #{slug.inspect}"
  warn "Use lowercase letters, numbers, and single hyphens."
  exit 1
end

if options[:categories].length > 2
  warn "Draft front matter supports up to two categories."
  exit 1
end

FileUtils.mkdir_p(options[:drafts_dir])
path = File.join(options[:drafts_dir], "#{slug}.md")

if File.exist?(path)
  warn "Refusing to overwrite existing draft: #{path}"
  exit 1
end

description = options[:description].to_s

content = <<~MARKDOWN
  ---
  layout: post
  title: "#{title}"
  description: "#{description}"
  image:
  categories:
  #{yaml_list(options[:categories], fallback_count: 2)}
  tags:
  #{yaml_list(options[:tags], fallback_count: 1)}
  math: true
  numbered_headings: true
  numbered_callouts: true
  numbered_equations: true
  ---

  > [!summary] Thesis
  > 

  ## Motivation

  ## Core Idea

  ## Outline

  ## Open Questions
MARKDOWN

File.write(path, content)
puts "Created #{path}"
