
require "digest"
require_relative "scholar_artifact_cache"

module Jekyll
  class Scholar
    module Utilities
      alias_method :old_cite, :cite
      alias_method :old_bibliography_tag, :bibliography_tag

      SCHOLAR_BACKLINKS_CACHE_VERSION = "2026-04-22-render-cache-v2".freeze

      def scholar_backlinks_config
        config['backlinks'] || { 'enabled' => false }
      end

      def scholar_site_cache(name)
        site.instance_variable_get(name) || site.instance_variable_set(name, {})
      end

      def shared_csl_renderer_cache
        scholar_site_cache(:@scholar_shared_csl_renderer_cache)
      end

      def converted_entry_cache
        scholar_site_cache(:@scholar_converted_entry_cache)
      end

      def scholar_bibliography_signature
        @scholar_bibliography_signature ||= begin
          paths = bibtex_paths
          Digest::SHA1.hexdigest(
            paths.sort.map { |path| "#{path}:#{File.mtime(path).to_f}" }.join("\n")
          )
        end
      end

      def converted_entry(entry)
        return unless entry
        return entry if bibtex_filters.empty?

        cache_key = [scholar_bibliography_signature, bibtex_filters.join(","), entry.key.to_s]
        converted_entry_cache[cache_key] ||= entry.dup.convert(*bibtex_filters)
      end

      def csl_renderer(force = false)
        cache_key = [style, config["locale"]]
        shared_csl_renderer_cache.delete(cache_key) if force
        shared_csl_renderer_cache[cache_key] ||= CiteProc::Ruby::Renderer.new(
          :format => "html",
          :style => style,
          :locale => config["locale"]
        )
      end

      def bibliography_template_source
        @bibliography_template_source ||= begin
          tmp = bibliography_template

          case
          when tmp.nil?
            ""
          when site.layouts.key?(tmp)
            site.layouts[tmp].content.to_s
          else
            tmp.to_s
          end
        end
      end

      def reference_only_bibliography_template?
        template = bibliography_template_source.strip
        template.empty? || template == "{{reference}}"
      end

      def reference_tag(entry, index = nil)
        return missing_reference unless entry

        entry = converted_entry(entry)
        reference = Jekyll::ScholarArtifactCache.artifact_reference_html(self, entry) || render_bibliography(entry, index)

        content_tag reference_tagname, reference,
          :id => [prefix, entry.key].compact.join('-')
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

            converted_entry(bibliography[key])
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
          bibliography_template_source,
          index,
          entry.key
        )

        cite_cache.getset(cache_key) do
          if reference_only_bibliography_template?
            reference_tag(entry, index)
          else
            old_bibliography_tag(entry, index)
          end
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
