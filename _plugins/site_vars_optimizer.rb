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

    def self.media_url(site, src, subpath: nil, absolute: false)
      url = src
      return url if url.nil? || url.include?(':')

      # 1. Add subpath
      url = File.join(subpath || '', url)
      # 2. Add CDN
      if site.config['cdn']
        url = File.join(site.config['cdn'], url)
      end
      # 3. Clean slashes
      url = url.gsub('///', '/').gsub('//', '/')
      
      return url if url.include?('://')

      # 4. Handle baseurl/url
      if absolute
        File.join(site.config['url'], site.baseurl, url)
      else
        File.join(site.baseurl, url)
      end
    end
    def self.render_seo(site, item)
      title = item.data['title'] || site.config['title']
      description = item.data['description'] || item.data['excerpt'] || site.config['description']
      author = item.data['author'] || site.config.dig('social', 'name')
      url = File.join(site.config['url'], site.baseurl, item.url).gsub(/\/$/, '/index.html').gsub(/\/$/, '')
      # Canonical URL fix
      canonical_url = File.join(site.config['url'], site.baseurl, item.url)

      html = []
      html << %(<meta name="generator" content="Jekyll v#{Jekyll::VERSION}" />)
      html << %(<meta property="og:title" content="#{title}" />)
      html << %(<meta name="author" content="#{author}" />)
      html << %(<meta property="og:locale" content="#{item.data['lang'] || 'en'}" />)
      html << %(<meta name="description" content="#{description}" />)
      html << %(<meta property="og:description" content="#{description}" />)
      html << %(<link rel="canonical" href="#{canonical_url}" />)
      html << %(<meta property="og:url" content="#{canonical_url}" />)
      html << %(<meta property="og:site_name" content="#{site.config['title']}" />)
      html << %(<meta property="og:type" content="#{item.data['layout'] == 'post' ? 'article' : 'website'}" />)
      html << %(<meta name="twitter:card" content="summary" />)
      html << %(<meta property="twitter:title" content="#{title}" />)
      if site.config.dig('twitter', 'username')
        html << %(<meta name="twitter:site" content="@#{site.config['twitter']['username']}" />)
        html << %(<meta name="twitter:creator" content="@#{site.config['twitter']['username']}" />)
      end

      html.join("\n")
    end
  end

  # Pre-calculate SEO and absolute image URLs to avoid expensive Liquid logic
  Hooks.register [:posts, :pages, :docs], :pre_render do |item, payload|
    site = item.site
    next unless item.output_ext == '.html'

    # 1. Fix page.image path to be absolute
    if item.data['image']
      src = item.data['image'].is_a?(Hash) ? item.data['image']['path'] : item.data['image']
      unless src.to_s.include?('://')
        abs_url = SiteVarsOptimizer.media_url(site, src, subpath: item.data['media_subpath'], absolute: true)
        if item.data['image'].is_a?(Hash)
          item.data['image']['path'] = abs_url
        else
          item.data['image'] = abs_url
        end
      end
    elsif site.config['social_preview_image']
      item.data['precomputed_social_image'] = SiteVarsOptimizer.media_url(site, site.config['social_preview_image'], absolute: true)
    end

    # 2. Pre-render SEO tags in Ruby natively to bypass Liquid overhead
    if Jekyll.env == 'production'
      item.data['precomputed_seo_html'] = SiteVarsOptimizer.render_seo(site, item)
    end
  end

  # High-performance Ruby-based HTML Compression
  # Replaces the theme's extremely slow Liquid-based compress.html layout
  Hooks.register [:posts, :pages, :docs], :post_render do |item|
    if Jekyll.env == 'production' && item.output_ext == '.html' && item.data['compress'] != false
      # Basic efficient whitespace removal
      # 1. Strip comments (optional, but good for size)
      item.output.gsub!(/<!--(?!\[if).*?-->/m, '')
      # 2. Collapse whitespace between tags
      item.output.gsub!(/>\s+</, '><')
      # 3. Trim leading/trailing whitespace
      item.output.strip!
    end
  end
end
