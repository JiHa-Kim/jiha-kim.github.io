
module Jekyll
  class Scholar
    module Utilities
      alias_method :old_cite, :cite
      alias_method :old_bibliography_tag, :bibliography_tag

      def scholar_backlinks_config
        config['backlinks'] || { 'enabled' => false }
      end

      # Cache rendered citations to avoid re-running expensive scholar logic
      def citation_cache
        @citation_cache ||= {}
      end

      def cite(keys)
        page = context.registers[:page]
        page['scholar_backlinks'] ||= Hash.new { |h, k| h[k] = [] }
        
        # Track citations and inject unique IDs for backtracking
        if config['separate_links']
          rendered_links = keys.map do |key|
            if bibliography.key?(key)
              entry = bibliography[key].dup
              entry = entry.convert(*bibtex_filters) unless bibtex_filters.empty?
              
              idx = page['scholar_backlinks'][key].length + 1
              backref_id = "cite-#{key}-#{idx}"
              page['scholar_backlinks'][key] << backref_id
              
              rendered = (citation_cache[key] ||= render_citation([entry])
                .sub(/^#{Regexp.escape(csl_prefix)}/, '')
                .sub(/#{Regexp.escape(csl_suffix)}$/, ''))
              
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
          backref_ids = []
          keys.each do |key|
            if bibliography.key?(key)
              idx = page['scholar_backlinks'][key].length + 1
              backref_id = "cite-#{key}-#{idx}"
              page['scholar_backlinks'][key] << backref_id
              backref_ids << backref_id
            end
          end
          
          primary_id = backref_ids.first
          
          if bibliography.key?(keys[0])
            items = keys.map { |k| bibliography[k] }.compact.map do |e|
              e = e.dup
              e = e.convert(*bibtex_filters) unless bibtex_filters.empty?
              e
            end
            cache_key = keys.join(',')
            rendered = (citation_cache[cache_key] ||= render_citation(items))

            link_to link_target_for(keys[0]), rendered, {
              class: config['cite_class'],
              id: primary_id
            }
          else
            missing_reference
          end
        end
      end

      def bibliography_tag(entry, index)
        reference = old_bibliography_tag(entry, index)
        
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
