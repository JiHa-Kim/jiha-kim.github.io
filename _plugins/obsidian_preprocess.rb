# frozen_string_literal: true

require 'jekyll'
require 'rexml/document'

module Jekyll
  class ObsidianPreprocess < Jekyll::Generator
    priority :high

    # Map Obsidian callout types -> Chirpy box classes
    TYPE_MAP = {
      # math family
      "definition" => "box-definition", "lemma" => "box-lemma", "proposition" => "box-proposition",
      "theorem" => "box-theorem", "example" => "box-example", "corollary" => "box-corollary",
      "remark" => "box-remark", "proof" => "box-proof", "principle" => "box-principle", "axiom" => "box-axiom",
      "postulate" => "box-postulate", "conjecture" => "box-conjecture", "claim" => "box-claim",
      "notation" => "box-notation", "algorithm" => "box-algorithm", "problem" => "box-problem",
      "exercise" => "box-exercise", "solution" => "box-solution", "assumption" => "box-assumption",
      "convention" => "box-convention", "fact" => "box-fact",
      # standard family
      "info" => "box-info", "note" => "box-note", "abstract" => "box-abstract", "summary" => "box-summary",
      "tldr" => "box-tldr", "todo" => "box-todo", "tip" => "box-tip", "hint" => "box-hint",
      "important" => "box-important", "success" => "box-success", "check" => "box-check", "done" => "box-done",
      "question" => "box-question", "help" => "box-help", "faq" => "box-faq",
      "warning" => "box-warning", "caution" => "box-caution", "attention" => "box-attention",
      "danger" => "box-danger", "error" => "box-error", "bug" => "box-bug", "failure" => "box-failure",
      "fail" => "box-fail", "missing" => "box-missing", "quote" => "box-quote", "cite" => "box-cite"
    }.freeze

    DEFAULT_OPEN_TYPES = %w[question help faq].freeze

    SKIP_FORMATTING_ENVS = %w[
      align align* equation equation* gather gather*
      multline multline* split cases dcases array
      matrix pmatrix bmatrix Bmatrix vmatrix Vmatrix
      aligned alignedat gathered subequations flalign
      flalign* eqnarray
    ].freeze

    PROTECT_RE = /((?:^\s*```.*?$)(?:.*?)(?:^\s*```$)|(?i:<div class="math-block"[^>]*>.*?<\/div>)|(?i:<span class="math-inline"[^>]*>.*?<\/span>)|(?:`[^`\n]*`))/m

    def generate(site)
      site.posts.docs.each { |doc| process_doc(doc) }
      site.pages.each { |page| process_doc(page) }
      site.collections.each do |_, collection|
        collection.docs.each { |doc| process_doc(doc) }
      end
    end

    def process_doc(doc)
      return unless doc.data['layout'] # Only process content with layouts
      return unless doc.content

      # Check if already processed (idempotency)
      return if doc.content.include?('class="math-block"') || doc.content.include?('class="math-inline"')

      doc.content = transform_markdown(doc.content)
    end

    def transform_markdown(content)
      buckets = []
      
      # 1. Protect code and existing HTML/Math regions
      content = content.gsub(PROTECT_RE) do |match|
        buckets << match
        "@@PROTECT#{buckets.length - 1}@@"
      end

      # 2. Convert Callouts
      content = convert_callouts(content)

      # 3. Convert Math
      content = convert_block_math(content)
      content = convert_inline_math(content)

      # 4. Restore Regions
      buckets.each_with_index do |payload, i|
        content.gsub!("@@PROTECT#{i}@@", payload)
      end

      content
    end

    def convert_callouts(md)
      lines = md.split("\n")
      out = []
      i = 0
      
      while i < lines.length
        line = lines[i]
        # Match > [!type][+-] [title] {.attr}
        if line.lstrip.start_with?(">") && line =~ /^\s*>\s*\[\!(?<typ>[A-Za-z0-9_-]+)\](?<state>[\+\-])?(?:\s+(?<title>.*?(?=\s+\{\.|$)))?(?:\s+\{\.(?<attr>[A-Za-z0-9_-]+)\})?\s*$/
          html, i2 = parse_callout_block(lines, i)
          if html
            out << html
            i = i2
            next
          end
        end
        out << line
        i += 1
      end
      out.join("\n")
    end

    def parse_callout_block(lines, start_idx)
      m = /^\s*>\s*\[\!(?<typ>[A-Za-z0-9_-]+)\](?<state>[\+\-])?(?:\s+(?<title>.*?(?=\s+\{\.|$)))?(?:\s+\{\.(?<attr>[A-Za-z0-9_-]+)\})?\s*$/.match(lines[start_idx])
      return nil unless m

      ctype = m[:typ].downcase
      state = m[:state]
      attr = m[:attr]
      title = (m[:title] || "").strip

      content_lines = []
      i = start_idx + 1
      
      in_code_fence = false
      in_math_block = false

      while i < lines.length
        line = lines[i]
        is_quoted = line.lstrip.start_with?(">")
        
        content_part = is_quoted ? line.sub(/^\s*>\s?/, '') : line
        
        keep_line = false
        if is_quoted
          keep_line = true
        elsif in_code_fence || in_math_block
          keep_line = true
        elsif line.strip.empty?
          # Blank line inside callout allowed ONLY if we are in a block (code or math)
          # In standard Markdown, a truly blank line breaks a blockquote.
          next_line = lines[i+1]
          if next_line && (in_code_fence || in_math_block)
             keep_line = true
          else
             break
          end
        else
          break
        end

        if keep_line
          content_lines << content_part
          in_code_fence = !in_code_fence if content_part.lstrip.start_with?("```") && !in_math_block
          in_math_block = !in_math_block if content_part.strip == "$$" && !in_code_fence
          i += 1
        else
          break
        end
      end

      # Recursively convert callouts in body
      raw_body = content_lines.join("\n")
      processed_body = convert_callouts(raw_body)
      # Robustly separate body from surrounding tags with double newlines
      processed_body = "\n\n" + processed_body.strip + "\n\n" unless processed_body.strip.empty?

      box_class = TYPE_MAP[ctype] || "box-info"
      box_class += " #{attr}" if attr

      is_collapsible = !state.nil? || DEFAULT_OPEN_TYPES.include?(ctype)
      is_open = (state == "+") || DEFAULT_OPEN_TYPES.include?(ctype)

      if is_collapsible
        open_attr = is_open ? " open" : ""
        summary_text = !title.empty? ? title : ctype.capitalize
        html = <<~HTML
          <details class="#{box_class}"#{open_attr} markdown="1">
          <summary markdown="1">
          #{summary_text}
          </summary>\n\n#{processed_body}</details>
        HTML
      else
        title_html = !title.empty? ? "<div class=\"title\" markdown=\"1\">\n#{title}\n</div>" : ""
        html = <<~HTML
          <blockquote class="#{box_class}" markdown="1">
          #{title_html}\n\n#{processed_body}</blockquote>
        HTML
      end

      [html.strip, i]
    end

    def convert_block_math(content)
      content.gsub(/(?m)(?:^([\t ]*))?(?:(?<!\\)(?<!\$)\$\$(?!\$)([\s\S]+?)(?<!\\)(?<!\$)\$\$(?!\$)|\\\[([\s\S]+?)\\\])/) do
        indent = $1 || ""
        math_content = $2 || $3
        
        has_env = SKIP_FORMATTING_ENVS.any? { |env| math_content.include?("\\begin{#{env}}") }
        math_content = cleanup_latex_syntax(math_content) unless has_env
        
        "\n#{indent}<div class=\"math-block\" markdown=\"0\">\n\\[\n#{math_content.strip}\n\\]\n#{indent}</div>\n"
      end
    end

    def convert_inline_math(content)
      content.gsub(/(?<!\\)(?<!\$)\$(?!\$)([^$\n]+?)(?<!\\)(?<!\$)\$(?!\$)|\\\(([\s\S]+?)\\\)/) do
        math_content = $1 || $2
        math_content = cleanup_latex_syntax(math_content)
        "<span class=\"math-inline\" markdown=\"0\">\\(#{math_content}\\)</span>"
      end
    end

    def cleanup_latex_syntax(text)
      text = text.gsub("...", "\\dots ")
      text = text.gsub(/\\\||\|\|/, "\\Vert ")
      # Match pipe NOT preceded by backslash
      text = text.gsub(/(?<!\\)\|/, "\\vert ")
      text = text.gsub("*", "\\ast ")
      text = text.gsub("~", "\\sim ")
      text
    end
  end
end
