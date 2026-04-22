# frozen_string_literal: true

require "digest"
require "jekyll"
require_relative "site_item_helpers"

module Jekyll
  # This plugin pre-calculates common variables that are otherwise
  # calculated repeatedly in Liquid templates (O(N) -> O(1)).
  class SiteVarsOptimizer < Generator
    priority :highest
    CACHE_VERSION = "2026-04-22-site-vars-v2".freeze

    def generate(site)
      @cache ||= Jekyll::Cache.new("SiteVarsOptimizer")

      site.data["origin_type"] = self.class.origin_type(site)
      site.data["prefer_mode"] = self.class.prefer_mode(site)
      site.data["favicon_path"] = self.class.favicon_path(site)

      Jekyll::SiteItemHelpers.each_unique_content_item(site) do |item|
        next unless item.respond_to?(:data)

        item.data["lang"] = self.class.resolve_lang(site, item)
        next unless item.respond_to?(:content) && item.content

        cache_key = Digest::MD5.hexdigest(
          "#{CACHE_VERSION}#{item.path}#{item.content}"
        )

        item.data["has_h2_h3"] = @cache.getset(cache_key) do
          self.class.has_h2_h3?(item.content)
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

    def self.origin_type(site)
      return "cors" unless site.config.dig("assets", "self_host", "enabled")

      env = site.config.dig("assets", "self_host", "env")
      return "basic" if env.nil? || env == Jekyll.env

      "cors"
    end

    def self.prefer_mode(site)
      mode = site.config["theme_mode"]
      mode ? %(data-mode="#{mode}") : ""
    end

    def self.favicon_path(site)
      File.join(site.baseurl || "", "/assets/img/favicons")
    end

    def self.resolve_lang(site, item)
      page_lang = item.data["lang"]
      return page_lang if page_lang && site.data.dig("locales", page_lang)
      return site.config["lang"] if site.data.dig("locales", site.config["lang"])

      "en"
    end

    def self.has_h2_h3?(content)
      return true if content.include?("<h2") || content.include?("<h3")
      return true if content.match?(/^\s{0,3}###?\s+\S/m)
      return true if content.match?(/^\S.+\n---+\s*$/m)

      false
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
      if item.data['precomputed_social_image']
        html << %(<meta property="og:image" content="#{item.data['precomputed_social_image']}" />)
        html << %(<meta name="twitter:card" content="summary_large_image" />)
        html << %(<meta property="twitter:image" content="#{item.data['precomputed_social_image']}" />)
      end

      html.join("\n")
    end
  end

  # Pre-calculate SEO and absolute image URLs to avoid expensive Liquid logic
  Hooks.register [:posts, :pages, :docs], :pre_render do |item, payload|
    site = item.site
    next unless item.output_ext == '.html'

    # 1. Precomputed SEO social image (Keep original item.data['image'] relative/intact)
    if item.data['image']
      src = item.data['image'].is_a?(Hash) ? item.data['image']['path'] : item.data['image']
      unless src.to_s.include?('://')
        item.data['precomputed_social_image'] = SiteVarsOptimizer.media_url(site, src, subpath: item.data['media_subpath'], absolute: true)
      end
    elsif site.config['social_preview_image']
      item.data['precomputed_social_image'] = SiteVarsOptimizer.media_url(site, site.config['social_preview_image'], absolute: true)
    end

    # 2. Pre-render SEO tags in Ruby natively to bypass Liquid overhead
    item.data['precomputed_seo_html'] = SiteVarsOptimizer.render_seo(site, item)
  end
  # High-performance Ruby-based HTML Compression
  # Replaces the theme's extremely slow Liquid-based compress.html layout
  Hooks.register [:posts, :pages, :docs], :post_render do |item|
    if Jekyll.env == 'production' && item.output_ext == '.html' && item.data['compress'] != false
      # Basic efficient whitespace removal
      # 1. Strip comments (optional, but good for size)
      # Preserve [if IE] style comments
      item.output.gsub!(/<!--(?!\[if).*?-->/m, '')
      # 2. Collapse whitespace between tags
      item.output.gsub!(/>\s+</, '><')
      # 3. Trim leading/trailing whitespace
      item.output.strip!
    end
  end
end
