# frozen_string_literal: true

require 'jekyll'
require 'rexml/document'
require 'digest'
require 'base64'

module Jekyll
  class ObsidianPreprocess < Jekyll::Generator
    priority :high
    CACHE_VERSION = "2026-04-28-algo-multiline-func-calls-v1".freeze

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
      "fail" => "box-fail", "missing" => "box-missing", "quote" => "box-quote", "cite" => "box-cite",
      "table" => "box-table"
    }.freeze

    LABELED_TYPES = %w[
      definition lemma proposition theorem example corollary remark proof
      principle axiom postulate conjecture claim notation algorithm problem
      exercise solution assumption convention fact caution table
    ].freeze

    DEFAULT_OPEN_TYPES = %w[question help faq].freeze

    SKIP_FORMATTING_ENVS = %w[
      align align* equation equation* gather gather*
      multline multline* split cases dcases array
      matrix pmatrix bmatrix Bmatrix vmatrix Vmatrix
      aligned alignedat gathered subequations flalign
      flalign* eqnarray
    ].freeze

    PROTECT_RE = /((?:^\s*```.*?$)(?:.*?)(?:^\s*```$)|(?i:<pre[^>]*>.*?<\/pre>)|(?i:<div class="math-block"[^>]*>.*?<\/div>)|(?i:<span class="math-inline"[^>]*>.*?<\/span>)|(?:(?<!`)`[^`\n]+`(?!`)))/m

    def generate(site)
      seen = {}

      site.pages.each { |page| process_doc(page, seen) }
      site.collections.each do |_, collection|
        collection.docs.each { |doc| process_doc(doc, seen) }
      end
    end

    def process_doc(doc, seen = nil)
      if seen
        return if seen[doc.object_id]
        seen[doc.object_id] = true
      end

      return if doc.data['obsidian_preprocessed']
      return unless doc.data['layout'] # Only process content with layouts
      return unless doc.content && !doc.content.empty?

      # Optimization: Skip if no Obsidian-style elements exist
      # This saves significant regex work on simple pages.
      has_algorithm = doc.content.include?("```") && doc.content.match?(/^\s*```(?:pseudo|pseudocode|algorithm)\b/m)
      return unless doc.content.include?('>') || doc.content.include?('$') || has_algorithm

      # Use Jekyll's persistent cache (stored in .jekyll-cache/)
      @cache ||= Jekyll::Cache.new("ObsidianPreprocess")
      
      # The cache key includes the content and layout/media_subpath which affect the transformation
      cache_key = Digest::MD5.hexdigest("#{CACHE_VERSION}#{doc.content}#{doc.data['layout']}#{doc.data['media_subpath']}")
      
      doc.content = @cache.getset(cache_key) do
        # Log only when actually transforming (cache miss)
        Jekyll.logger.debug "ObsidianPreprocess:", "Cache miss for #{doc.relative_path}, transforming..."
        transform_markdown(doc.content, doc)
      end

      doc.data['obsidian_preprocessed'] = true
    end

    def transform_markdown(content, doc)
      # 1. Convert Algorithms (must run before protection hides <pre> blocks)
      content = convert_algorithms(content)

      buckets = []
      
      # 2. Protect code and existing HTML/Math regions
      content = content.gsub(PROTECT_RE) do |match|
        buckets << match
        "@@PROTECT#{buckets.length - 1}@@"
      end

      # 3. Convert Callouts
      content = convert_callouts(content)

      # 4. Convert Math
      content = convert_block_math(content)
      content = convert_inline_math(content)

      # 5. Restore Regions
      buckets.each_with_index do |payload, i|
        content.gsub!("@@PROTECT#{i}@@") { payload }
      end

      # 6. Preserve spaces between tokens at risk of being stripped by compress_html
      content = preserve_semantic_spaces(content)

      content
    end

    def preserve_semantic_spaces(content)
      return content unless content.include?('</span>') || content.include?('$')

      # Protect custom-algo blocks from space injection (avoids flex-token justification)
      algo_buckets = []
      content = content.gsub(/<div class="custom-algo".*?<\/div>/m) do |match|
        algo_buckets << match
        "__ALGO_BLOCK_#{algo_buckets.length - 1}__"
      end

      # Identify spaces between adjacent elements that should not be stripped
      # 1. Between two math spans
      content = content.gsub(/<\/span>\s+<span class="math-inline"/, '</span>&nbsp;<span class="math-inline"')
      # 2. Between math and algo spans
      content = content.gsub(/<\/span>\s+(?=<span class="algo-)/, '</span>&nbsp;')
      content = content.gsub(/(?<=<\/span>)\s+<span class="math-inline"/, '&nbsp;<span class="math-inline"')
      # 3. Between math symbols and spans
      content = content.gsub(/<\/span>\s+(?=\$)/, '</span>&nbsp;')
      content = content.gsub(/(?<=\$)\s+<span/, '&nbsp;<span')
      # 4. Between adjacent math symbols
      content = content.gsub(/(?<=\$)\s+(?=\$)/, '&nbsp;')

      algo_buckets.each_with_index do |payload, i|
        content.gsub!("__ALGO_BLOCK_#{i}__") { payload }
      end

      content
    end

    def refactor_tables(content)
      return content unless content.include?('<table')
      # Wrap tables except those already inside a code block
      content.gsub(/<table.*?<\/table>/m) do |table|
        "<div class=\"table-wrapper\">#{table}</div>"
      end
    end

    def refactor_checkboxes(content)
      return content unless content.include?('type="checkbox"')
      content.gsub(/<input type="checkbox" class="task-list-item-checkbox" disabled="disabled" checked="checked" \/>/, '<i class="fas fa-check-circle fa-fw checked"></i>')
             .gsub(/<input type="checkbox" class="task-list-item-checkbox" disabled="disabled" \/>/, '<i class="far fa-circle fa-fw"></i>')
    end

    def refactor_headings(content)
      # Process h2, h3, h4, h5
      content.gsub(/<h([2-5]) id="([^"]+)">(.+?)<\/h\1>/) do
        level = $1
        id = $2
        text = $3
        anchor = "<a href=\"##{id}\" class=\"anchor text-muted\"><i class=\"fas fa-hashtag\"></i></a>"
        "<h#{level} id=\"#{id}\"><span class=\"me-2\">#{text}</span>#{anchor}</h#{level}>"
      end
    end

    def refactor_images(content, doc)
      return content unless content.include?('<img ')
      media_subpath = doc.data['media_subpath'] || ""
      
      content.gsub(/<img (.*?)>/) do |match|
        attrs_str = $1
        attrs = parse_html_attrs(attrs_str)
        
        src = attrs['src']
        next match unless src # Safety
        
        # Handle media_subpath
        unless src.start_with?('http', '//', '/')
          src = File.join('/', media_subpath, src).gsub(/\/+/, '/')
        end
        
        # Apply lazy loading and shimmer
        classes = (attrs['class'] || "").split(' ')
        classes << 'shimmer' unless attrs['lqip']
        
        new_attrs = attrs.dup
        new_attrs['src'] = src
        new_attrs['loading'] = 'lazy'
        new_attrs['class'] = classes.join(' ')
        
        img_tag = "<img #{serialize_attrs(new_attrs)}>"
        
        # Wrap in popup link if not on home page
        if doc.data['layout'] == 'home'
          "<div class=\"preview-img #{new_attrs['class']}\">#{img_tag}</div>"
        else
          "<a href=\"#{src}\" class=\"popup img-link #{new_attrs['class']}\">#{img_tag}</a>"
        end
      end
    end

    def parse_html_attrs(str)
      attrs = {}
      str.scan(/(\w+)="([^"]*)"/).each do |k, v|
        attrs[k] = v
      end
      attrs
    end

    def serialize_attrs(attrs)
      attrs.map { |k, v| "#{k}=\"#{v}\"" }.join(' ')
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

      # Smart joiner that injects extra newlines for block-level elements
      joined_body = String.new
      in_code = false
      in_math = false
      content_lines.each_with_index do |line, idx|
        joined_body << line << "\n"
        in_code = !in_code if line.lstrip.start_with?("```")
        in_math = !in_math if line.strip == "$$" && !in_code
        # puts "DEBUG: line='#{line}', in_code=#{in_code}"
        
        next_line = content_lines[idx+1]
        if next_line && !in_code && !in_math && !line.strip.empty? && !next_line.strip.empty?
          # Do not split table rows (lines containing '|' with structure)
          # A line is a table row if it contains a pipe and neighbors also contain pipes
          # OR if it's a separator line like |--|
          is_table_row = (line.include?('|') && next_line.include?('|')) || 
                         (next_line.strip =~ /^\|?[\s\-\:\|]+\|?$/)
          
          unless is_table_row
            # In these technical posts, treat every line break as a paragraph break
            # to ensure math blocks and comments stay separated.
            joined_body << "\n"
          end
        end
      end

      # Recursively convert callouts in body
      processed_body = convert_callouts(joined_body)

      # Robustly separate body from surrounding tags with double newlines
      processed_body = "\n\n" + processed_body.strip + "\n\n" unless processed_body.strip.empty?




      box_class = TYPE_MAP[ctype] || "box-info"
      box_class += " #{attr}" if attr

      is_collapsible = !state.nil? || DEFAULT_OPEN_TYPES.include?(ctype)
      is_open = (state == "+") || DEFAULT_OPEN_TYPES.include?(ctype)

      effective_title = !title.empty? ? title : (LABELED_TYPES.include?(ctype) ? "&nbsp;" : ctype.capitalize)

      if is_collapsible
        open_attr = is_open ? " open" : ""
        html = <<~HTML
          <details class="#{box_class}"#{open_attr} markdown="1">
          <summary markdown="1">
          #{effective_title}
          </summary>\n\n#{processed_body}</details>
        HTML
      else
        html = <<~HTML
          <blockquote class="#{box_class}" markdown="1">
          <div class="title" markdown="1">
          #{effective_title}
          </div>\n\n#{processed_body}</blockquote>
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

        encoded_source = encode_math_source(math_content.strip)

        "\n#{indent}<div class=\"math-block\" data-math-source-b64=\"#{encoded_source}\" markdown=\"0\">\n\\[\n#{math_content.strip}\n\\]\n#{indent}</div>\n"
      end
    end

    def convert_inline_math(content)
      content.gsub(/(?<!\\)\$(?!\$)([^$\n]+?)(?<!\\)\$|\\\(([\s\S]+?)\\\)/) do
        math_content = $1 || $2
        math_content = cleanup_latex_syntax(math_content)
        encoded_source = encode_math_source(math_content.strip)
        "<span class=\"math-inline\" data-math-source-b64=\"#{encoded_source}\" markdown=\"0\">\\(#{math_content.strip}\\)</span>"
      end
    end

    def encode_math_source(text)
      Base64.strict_encode64(text.to_s)
    end

    def cleanup_latex_syntax(text)
      text = text.gsub("...", "\\dots ")
      text = text.gsub(/\\\||\|\|/, "\\Vert ")
      # Match pipe NOT preceded by backslash
      text = text.gsub(/(?<!\\)\|/, "\\vert ")
      text = text.gsub("*", "\\ast ")
      text = text.gsub("~", "\\sim ")
      # Protect brackets from Markdown link detection
      text = text.gsub("[", "\\lbrack ")
      text = text.gsub("]", "\\rbrack ")
      text
    end
    def convert_algorithms(md)
      md.gsub(/^```(?:pseudo|pseudocode|algorithm)\b[^\n]*\n(.*?)^```$\n?/m) do
        html = parse_pythonic_pseudocode($1)
        render_algorithm_block(html)
      end
    end

    def render_algorithm_block(html)
      "<div class=\"custom-algo\" markdown=\"0\">\n#{html}\n</div>"
    end

    def parse_pythonic_pseudocode(text)
      output = []
      line_num = 1

      text.split("\n").each do |line|
        raw = line.rstrip
        next if raw.strip.empty?

        indent = raw[/\A[ \t]*/].to_s.gsub("\t", "    ").length / 4
        processed_line = process_pythonic_algo_line(raw.strip)

        output << "<div class=\"algo-line\" style=\"--indent: #{indent};\">"
        output << "  <span class=\"algo-linenum\">#{line_num}:</span>"
        output << "  <span class=\"algo-content\">#{processed_line}</span>"
        output << "</div>"

        line_num += 1
      end

      output.join("\n")
    end

    def process_pythonic_algo_line(line)
      code, comment = line.split(/(?<!\\)#/, 2)
      code = highlight_pythonic_code(code.rstrip)
      comment_html = comment ? "<span class=\"algo-comment\">&nbsp;# #{comment.strip}</span>" : ""
      [code, comment_html].reject(&:empty?).join("")
    end

    def highlight_pythonic_code(code)
      code, buckets = protect_algo_math(code)

      # Handle multiline calls before the balanced-call matcher sees the bare name only.
      code = code.gsub(/@([A-Za-z_]\w+)\s*\(\s*\z/) do
        "<span class=\"algo-func\">#{$1}</span><span class=\"algo-func paren\">(</span>"
      end

      # Handle @Func(...) and def Func(...) including parentheses and LaTeX scaling.
      # Uses a recursive regex for balanced parentheses: (\((?>[^()]+|\g<2>)*\))
      code = code.gsub(/@([A-Za-z_]\w+)(?:\s*(\((?>[^()]+|\g<2>)*\)))?/) do
        format_algo_func($1, $2, buckets)
      end

      code = code.gsub(/\b(def)\s+([A-Za-z_]\w+)(?:\s*(\((?>[^()]+|\g<3>)*\)))?/) do
        "<span class=\"algo-kw\">#{$1}</span>&nbsp;" + format_algo_func($2, $3, buckets)
      end

      %w[if elif else for while return break continue try except finally with pass yield until in].each do |kw|
        code = code.gsub(/\b#{Regexp.escape(kw)}\b/, "<span class=\"algo-kw\">#{kw}</span>")
      end

      code = highlight_algo_literals(code)
      code = code.gsub(/:/, "<span class=\"algo-punct\">:</span>")

      code = restore_algo_math(code, buckets)

      # FIX: Join adjacent math blocks (e.g., $A$$B$ -> $AB$) 
      # This fixes both MathJax superscript context and the site-wide block-math misinterpretation bug.
      # We do it after restoration but before returning to the global pipeline.
      code = code.gsub(/\$\$/, '')
      code = code.gsub(/\A\)\z/, "<span class=\"algo-func paren\">)</span>")

      code
    end

    def format_algo_func(name, args, buckets)
      if args
        # Strip potential wrapping parentheses from the recursive regex match
        args = args[1..-2] if args.start_with?("(") && args.end_with?(")")

        # If args is a single math block, move parentheses inside and use \left/\right
        if args.strip =~ /^(__ALGOMATH_(\d+)__)$/
          idx = $2.to_i
          math_content = buckets[idx]
          if math_content =~ /^\$(.*)\$$/
            inner = $1.strip
            buckets[idx] = "$\\left( #{inner} \\right)$"
            return "<span class=\"algo-func\">#{name}</span>#{args}"
          end
        end
        # Fallback for mixed/other content or non-math
        "<span class=\"algo-func\">#{name}</span><span class=\"algo-func paren\">(</span>#{args}<span class=\"algo-func paren\">)</span>"
      else
        "<span class=\"algo-func\">#{name}</span>"
      end
    end

    def highlight_algo_literals(line)
      token_map = {
        "true" => "algo-literal",
        "false" => "algo-literal",
        "none" => "algo-literal",
        "success" => "algo-literal",
        "failure" => "algo-literal"
      }

      token_map.each do |word, css_class|
        line = line.gsub(/\b#{Regexp.escape(word)}\b/i) do
          "<span class=\"#{css_class}\">#{$&}</span>"
        end
      end

      line
    end

    def protect_algo_math(text)
      buckets = []
      protected = text.gsub(/\$\$.*?\$\$|\$[^$\n]+\$/m) do |match|
        buckets << match
        "__ALGOMATH_#{buckets.length - 1}__"
      end
      [protected, buckets]
    end

    def restore_algo_math(text, buckets)
      buckets.each_with_index do |payload, i|
        text = text.gsub("__ALGOMATH_#{i}__") { payload }
      end
      text
    end
  end
end

# Register post_convert hook for HTML refactorings
OBSIDIAN_HTML_REFACTORER = Jekyll::ObsidianPreprocess.new(nil)

Jekyll::Hooks.register [:pages, :documents], :post_convert do |doc|
  # Only process full HTML content docs
  next unless doc.content&.include?("<")

  content = doc.content

  if content.include?("<table")
    content = OBSIDIAN_HTML_REFACTORER.refactor_tables(content)
  end

  if content.include?('type="checkbox"')
    content = OBSIDIAN_HTML_REFACTORER.refactor_checkboxes(content)
  end

  if content.include?("<img ")
    content = OBSIDIAN_HTML_REFACTORER.refactor_images(content, doc)
  end

  if content.include?("<h2") || content.include?("<h3") || content.include?("<h4") || content.include?("<h5")
    content = OBSIDIAN_HTML_REFACTORER.refactor_headings(content)
  end

  doc.content = content
end
