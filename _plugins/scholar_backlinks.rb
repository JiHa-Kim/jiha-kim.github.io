
module Jekyll
  class Scholar
    module Utilities
      alias_method :old_cite, :cite
      alias_method :old_bibliography_tag, :bibliography_tag

      def scholar_backlinks_config
        config['backlinks'] || { 'enabled' => false }
      end

      def cite(keys)
        # Ensure we have a place to store our back-references
        page = context.registers[:page]
        page['scholar_backlinks'] ||= Hash.new { |h, k| h[k] = [] }
        
        # We need to render each key to get its link, 
        # but we also need to add a unique ID to the citation link itself.
        
        # If multiple keys are cited together, they share the same link in the default configuration
        # (unless separate_links is true).
        
        if config['separate_links']
          # Handle separate links case
          rendered_links = keys.map do |key|
            if bibliography.key?(key)
              entry = bibliography[key]
              
              # Track this citation occurrence
              idx = page['scholar_backlinks'][key].length + 1
              backref_id = "cite-#{key}-#{idx}"
              page['scholar_backlinks'][key] << backref_id
              
              # Render the citation text
              rendered = render_citation([entry])
                .sub(/^#{Regexp.escape(csl_prefix)}/, '')
                .sub(/#{Regexp.escape(csl_suffix)}$/, '')
              
              # Render with the unique ID for backtracking
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
          # Handle combined links case (default)
          backref_ids = []
          keys.each do |key|
            if bibliography.key?(key)
              idx = page['scholar_backlinks'][key].length + 1
              backref_id = "cite-#{key}-#{idx}"
              page['scholar_backlinks'][key] << backref_id
              backref_ids << backref_id
            end
          end
          
          # Use the first backref_id for the group's ID
          primary_id = backref_ids.first
          
          if bibliography.key?(keys[0])
            items = keys.map { |k| bibliography[k] }.compact
            link_to link_target_for(keys[0]), render_citation(items), {
              class: config['cite_class'],
              id: primary_id
            }
          else
            missing_reference
          end
        end
      end

      def bibliography_tag(entry, index)
        # Render the standard bibliography entry
        reference = old_bibliography_tag(entry, index)
        
        # Append backlinks if they exist and are enabled
        return reference unless scholar_backlinks_config['enabled']

        page = context.registers[:page]
        backlinks = page['scholar_backlinks'] && page['scholar_backlinks'][entry.key]
        
        if backlinks && !backlinks.empty?
          symbol = scholar_backlinks_config['symbol'] || "&#8617;&#xfe0e;"
          css_class = scholar_backlinks_config['class'] || "scholar-backlink"

          links_html = backlinks.each_with_index.map do |id, i|
            # Match the footnote style of the theme if possible
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
