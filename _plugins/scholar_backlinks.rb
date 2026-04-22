
require "digest"

module Jekyll
  class Scholar
    module Utilities
      alias_method :old_cite, :cite
      alias_method :old_bibliography_tag, :bibliography_tag

      SCHOLAR_BACKLINKS_CACHE_VERSION = "2026-04-22-render-cache-v1".freeze

      def scholar_backlinks_config
        config['backlinks'] || { 'enabled' => false }
      end

      def scholar_render_cache_key(*parts)
        digest = Digest::SHA1.hexdigest(parts.flatten.compact.map(&:to_s).join("\0"))
        "scholar-backlinks:#{SCHOLAR_BACKLINKS_CACHE_VERSION}:#{digest}"
      end

      def rendered_citation(keys, strip_wrappers: false)
        cache_key = scholar_render_cache_key(
          "citation",
          style,
          config["locale"],
          bibtex_filters.join(","),
          keys
        )

        cite_cache.getset(cache_key) do
          items = keys.filter_map do |key|
            next unless bibliography.key?(key)

            entry = bibliography[key].dup
            entry = entry.convert(*bibtex_filters) unless bibtex_filters.empty?
            entry
          end

          rendered = render_citation(items)
          if strip_wrappers
            rendered = rendered.sub(/^#{Regexp.escape(csl_prefix)}/, "")
            rendered = rendered.sub(/#{Regexp.escape(csl_suffix)}$/, "")
          end
          rendered
        end
      end

      def cached_bibliography_reference(entry, index)
        page = context.registers[:page] || {}
        cache_key = scholar_render_cache_key(
          "bibliography",
          style,
          config["locale"],
          page["path"] || page["url"] || page["id"],
          index,
          entry.key
        )

        cite_cache.getset(cache_key) do
          old_bibliography_tag(entry, index)
        end
      end

      def cite(keys)
        page = context.registers[:page]
        page['scholar_backlinks'] ||= Hash.new { |h, k| h[k] = [] }
        
        # Track citations and inject unique IDs for backtracking
        if config['separate_links']
          rendered_links = keys.map do |key|
            if bibliography.key?(key)
              idx = page['scholar_backlinks'][key].length + 1
              backref_id = "cite-#{key}-#{idx}"
              page['scholar_backlinks'][key] << backref_id

              rendered = rendered_citation([key], strip_wrappers: true)

              link_to link_target_for(key), rendered, {
                class: config['cite_class'],
                id: backref_id
              }
            else
              missing_reference
            end
          end
          
          csl_prefix + rendered_links.join(delimiter) + csl_suffix
        else
          # Default case: combined links
          valid_keys = keys.select { |key| bibliography.key?(key) }
          primary_key = valid_keys.first

          if primary_key
            primary_idx = page['scholar_backlinks'][primary_key].length + 1
            primary_id = "cite-#{primary_key}-#{primary_idx}"

            # A combined citation renders as one HTML anchor, so every cited
            # entry must backlink to that shared anchor instead of unique ids
            # that do not exist in the output.
            valid_keys.each do |key|
              page['scholar_backlinks'][key] << primary_id
            end

            rendered = rendered_citation(valid_keys)

            link_to link_target_for(primary_key), rendered, {
              class: config['cite_class'],
              id: primary_id
            }
          else
            missing_reference
          end
        end
      end

      def bibliography_tag(entry, index)
        reference = cached_bibliography_reference(entry, index)
        
        # Append back-references if enabled
        return reference unless scholar_backlinks_config['enabled']

        page = context.registers[:page]
        backlinks = page['scholar_backlinks'] && page['scholar_backlinks'][entry.key]
        
        if backlinks && !backlinks.empty?
          symbol = scholar_backlinks_config['symbol'] || "&#8617;&#xfe0e;"
          css_class = scholar_backlinks_config['class'] || "scholar-backlink"

          links_html = backlinks.each_with_index.map do |id, i|
            label = backlinks.length > 1 ? (i + 1).to_s : symbol
            "<a href=\"##{id}\" class=\"#{css_class}\">#{label}</a>"
          end.join(" ")
          
          reference + " <span class=\"scholar-backlinks\">#{links_html}</span>"
        else
          reference
        end
      end
    end
  end
end
