#!/usr/bin/env ruby
# frozen_string_literal: true

require "set"
require "yaml"
require "date"

paths = ARGV

if paths.empty? || paths.include?("-h") || paths.include?("--help")
  puts "Usage: ruby tools/lint_markdown_automation.rb FILE [FILE ...]"
  puts
  puts "Checks Markdown automation conventions:"
  puts "  - manual heading numbers with numbered_headings: true"
  puts "  - equation references without matching {#eq:...} labels"
  puts "  - equation labels/references without numbered_equations: true"
  puts "  - raw ChatGPT-style math delimiters [ and ] on their own lines"
  puts "  - manual HTML callout tags missing markdown=\"1\""
  exit(paths.empty? ? 1 : 0)
end

MATH_CALLOUT_TYPES = %w[
  definition lemma proposition theorem example corollary remark proof
  principle axiom postulate conjecture claim notation algorithm problem
  exercise solution assumption convention fact
].freeze

Issue = Struct.new(:path, :line, :message) do
  def to_s
    "#{path}:#{line}: #{message}"
  end
end

def front_matter_for(text)
  return {} unless text.start_with?("---\n")

  parts = text.split(/^---\s*$/, 3)
  return {} unless parts.length >= 3

  YAML.safe_load(parts[1], permitted_classes: [Time, Date], aliases: true) || {}
rescue Psych::SyntaxError
  {}
end

def truthy?(value)
  value == true || value.to_s.downcase == "true"
end

def protected_code_lines(lines)
  protected = Set.new
  in_code = false

  lines.each_with_index do |line, idx|
    if line.lstrip.start_with?("```")
      protected << idx + 1
      in_code = !in_code
      next
    end

    protected << idx + 1 if in_code
  end

  protected
end

issues = []

paths.each do |path|
  unless File.file?(path)
    issues << Issue.new(path, 0, "file not found")
    next
  end

  text = File.read(path)
  data = front_matter_for(text)
  numbered_headings = truthy?(data["numbered_headings"] || data[:numbered_headings])
  numbered_equations = truthy?(data["numbered_equations"] || data[:numbered_equations])

  labels = text.scan(/\{#(eq:[A-Za-z0-9_.:-]+)\}/).flatten.to_set
  refs = text.scan(/(?<![\w.-])@(eq:[A-Za-z0-9_.:-]+)/).flatten

  if labels.any? && !numbered_equations
    issues << Issue.new(path, 1, "equation labels require numbered_equations: true")
  end

  if refs.any? && !numbered_equations
    issues << Issue.new(path, 1, "equation references require numbered_equations: true")
  end

  refs.each do |ref|
    next if labels.include?(ref)

    line = text[0...text.index("@#{ref}")].count("\n") + 1
    issues << Issue.new(path, line, "unknown equation reference @#{ref}")
  end

  lines = text.lines
  protected = protected_code_lines(lines)

  lines.each_with_index do |line, idx|
    line_no = idx + 1
    next if protected.include?(line_no)

    if numbered_headings && line =~ /^\s*\#{2,5}\s+\d+(?:\.\d+)*[.)]?\s+\S/
      issues << Issue.new(path, line_no, "remove manual heading number when numbered_headings: true is enabled")
    end

    if line.strip == "[" || line.strip == "]"
      issues << Issue.new(path, line_no, "use $$ display math delimiters instead of bare [ or ]")
    end

    if line =~ /<(?<tag>blockquote|details)\b/i && line !~ /markdown\s*=\s*["']1["']/i
      issues << Issue.new(path, line_no, "manual HTML #{Regexp.last_match[:tag]} should include markdown=\"1\"")
    end

    next unless line =~ /^\s*>\s*\[\!(?<type>[A-Za-z0-9_-]+)\]\s*$/

    type = Regexp.last_match[:type].downcase
    if MATH_CALLOUT_TYPES.include?(type)
      issues << Issue.new(path, line_no, "math callout [!#{type}] should usually have a title")
    end
  end
end

if issues.empty?
  puts "No Markdown automation issues found."
  exit 0
end

issues.each { |issue| warn issue }
exit 1
