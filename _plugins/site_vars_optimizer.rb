# frozen_string_literal: true

module Jekyll
  # This plugin pre-calculates common variables that are otherwise
  # calculated repeatedly in Liquid templates (O(N) -> O(1)).
  class SiteVarsOptimizer < Generator
    priority :highest

    def generate(site)
      # 1. Pre-calculate Site Origin Type
      origin_type = 'cors'
      if site.config.dig('assets', 'self_host', 'enabled')
        env = site.config.dig('assets', 'self_host', 'env')
        if env.nil? || env == Jekyll.env
          origin_type = 'basic'
        end
      end
      site.data['origin_type'] = origin_type

      # 2. Pre-calculate Site Preference Mode
      site.data['prefer_mode'] = ""
      if site.config['theme_mode']
        site.data['prefer_mode'] = %(data-mode="#{site.config['theme_mode']}")
      end

      # 3. Pre-calculate Document-specific variables
      site.each_site_file do |item|
        next unless item.respond_to?(:data)
        
        # Language detection
        page_lang = item.data['lang']
        if page_lang && site.data.dig('locales', page_lang)
          item.data['lang'] = page_lang
        elsif site.data.dig('locales', site.config['lang'])
          item.data['lang'] = site.config['lang']
        else
          item.data['lang'] = 'en'
        end

        # Header detection for TOC/Math (avoiding expensive contains checks in Liquid)
        if item.respond_to?(:content) && item.content
          item.data['has_h2_h3'] = item.content.include?('<h2') || item.content.include?('<h3')
        end
      end
    end
  end
end
