# frozen_string_literal: true

module Jekyll
  # This plugin pre-calculates common variables that are otherwise
  # calculated repeatedly in Liquid templates (O(N) -> O(1)).
  class SiteVarsOptimizer < Generator
    priority :highest

    def generate(site)
      # 1. Pre-calculate Site Origin Type
      # Mirroring logic from _includes/origin-type.html
      origin_type = 'cors'
      if site.config.dig('assets', 'self_host', 'enabled')
        env = site.config.dig('assets', 'self_host', 'env')
        if env.nil? || env == Jekyll.env
          origin_type = 'basic'
        end
      end
      site.data['origin_type'] = origin_type

      # 2. Pre-calculate Language for all documents
      # Mirroring logic from _includes/lang.html
      site.each_site_file do |item|
        next unless item.respond_to?(:data)
        
        page_lang = item.data['lang']
        if page_lang && site.data.dig('locales', page_lang)
          item.data['lang'] = page_lang
        elsif site.data.dig('locales', site.config['lang'])
          item.data['lang'] = site.config['lang']
        else
          item.data['lang'] = 'en'
        end
      end
    end
  end
end
