#!/usr/bin/env ruby
# frozen_string_literal: true

require "minitest/autorun"
require_relative "../_plugins/obsidian_preprocess"

Doc = Struct.new(:data, :relative_path)

class MarkdownAutomationTest < Minitest::Test
  def setup
    @processor = Jekyll::ObsidianPreprocess.new(nil)
  end

  def doc(options = {})
    Doc.new({ "layout" => "post" }.merge(options), "test.md")
  end

  def test_heading_numbering_nested_levels
    html = <<~HTML
      <h2 id="a">A</h2>
      <h3 id="b">B</h3>
      <h4 id="c">C</h4>
    HTML

    numbered = @processor.number_headings(html, doc("numbered_headings" => true))

    assert_includes numbered, '<span class="heading-number">1</span> A'
    assert_includes numbered, '<span class="heading-number">1.1</span> B'
    assert_includes numbered, '<span class="heading-number">1.1.1</span> C'
  end

  def test_callout_numbering_uses_current_section
    markdown = <<~MARKDOWN
      ## Geometry

      > [!definition] Metric
      > A distance-like object.
    MARKDOWN

    rendered = @processor.transform_markdown(markdown, doc("numbered_callouts" => true))

    assert_includes rendered, "box-definition box-numbered"
    assert_includes rendered, "Definition 1.1. Metric"
  end

  def test_equation_numbering_is_section_aware_and_references_resolve
    markdown = <<~MARKDOWN
      ## First

      $$
      {#eq:first}
      a=b
      $$

      See @eq:first.

      ## Second

      $$
      {#eq:second}
      c=d
      $$

      See @eq:second.
    MARKDOWN

    rendered = @processor.transform_markdown(markdown, doc("numbered_equations" => true))

    assert_includes rendered, 'data-eq-number="1.1"'
    assert_includes rendered, 'data-eq-number="2.1"'
    assert_includes rendered, 'href="#eq-first" class="eq-ref">Equation&nbsp;(1.1)</a>'
    assert_includes rendered, 'href="#eq-second" class="eq-ref">Equation&nbsp;(2.1)</a>'
  end

  def test_section_references_resolve_to_numbered_headings
    html = <<~HTML
      <h2 id="main-result"><span class="heading-number">3</span> Main Result</h2>
      <p>See @sec:main-result.</p>
    HTML

    rendered = @processor.convert_section_references(html, doc("numbered_headings" => true))

    assert_includes rendered, 'href="#main-result" class="sec-ref">Section&nbsp;3</a>'
  end
end
