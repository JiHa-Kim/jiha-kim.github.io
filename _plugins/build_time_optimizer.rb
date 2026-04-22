# frozen_string_literal: true

require "cgi"
require "digest"
require "json"
require "jekyll"
require "liquid"
require "time"
require "addressable/uri"
require_relative "site_item_helpers"

module Jekyll
  class BuildTimeOptimizer < Generator
    priority :lowest

    CACHE_VERSION = "2026-04-22-plain-text-v1".freeze
    WORDS_PER_MINUTE = 180
    MIN_READ_TIME = 1
    MAX_TRENDING_TAGS = 10
    SITEMAP_STATIC_FILE_EXTENSIONS = %w(.htm .html .xhtml .pdf).freeze
    SITEMAP_URLSET_OPEN = '<urlset xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://www.sitemaps.org/schemas/sitemap/0.9 http://www.sitemaps.org/schemas/sitemap/0.9/sitemap.xsd" xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'.freeze

    def generate(site)
      @cache ||= Jekyll::Cache.new("BuildTimeOptimizer")

      precompute_trending_tags(site)
      precompute_plain_text(site)
      precompute_search_index(site)
      precompute_sitemap(site)
    end

    private

    def precompute_trending_tags(site)
      site.data["trending_tags"] = site.tags
        .map { |name, docs| [name, docs.size] }
        .sort_by { |name, size| [-size, name.to_s.downcase] }
        .first(MAX_TRENDING_TAGS)
        .map(&:first)
    end

    def precompute_plain_text(site)
      Jekyll::SiteItemHelpers.each_post_like_item(site) do |item|
        next unless item.content && !item.content.empty?

        cache_key = Digest::MD5.hexdigest("#{CACHE_VERSION}#{item.content}")
        cached = @cache.getset(cache_key) do
          plain_text = self.class.plain_text(item.content)
          words = self.class.count_words(plain_text)

          {
            "plain_text" => plain_text,
            "readtime_words" => words,
            "readtime_minutes" => [words / WORDS_PER_MINUTE, MIN_READ_TIME].max
          }
        end

        item.data["plain_text_content"] = cached["plain_text"]
        item.data["readtime_words"] = cached["readtime_words"]
        item.data["readtime_minutes"] = cached["readtime_minutes"]
      end
    end

    def precompute_search_index(site)
      entries = site.posts.docs.map do |post|
        {
          "title" => post.data["title"],
          "url" => join_url(site.baseurl, post.url),
          "categories" => Array(post.data["categories"]).join(", "),
          "tags" => Array(post.data["tags"]).join(", "),
          "date" => format_date(post.data["date"]),
          "content" => post.data["plain_text_content"].to_s
        }
      end

      site.data["search_index_json"] = JSON.generate(entries)
    end

    def precompute_sitemap(site)
      rows = []

      site.collections.values.sort_by(&:label).each do |collection|
        next unless collection.write?

        collection.docs.each do |doc|
          next if doc.data["sitemap"] == false

          rows << sitemap_url_row(site, doc.url, doc.data["last_modified_at"] || doc.date)
        end
      end

      site.pages.each do |page|
        next unless page.html? || page.url.end_with?("/")
        next if page.data["sitemap"] == false
        next if page.url == "/404.html"

        rows << sitemap_url_row(site, page.url, page.data["last_modified_at"])
      end

      site.static_files.each do |file|
        next unless SITEMAP_STATIC_FILE_EXTENSIONS.include?(file.extname)
        next if file.name == "404.html"
        next if file.respond_to?(:data) && file.data["sitemap"] == false

        rows << sitemap_url_row(site, file.relative_path, file.modified_time)
      end

      site.data["sitemap_xml"] = [
        %(<?xml version="1.0" encoding="UTF-8"?>),
        SITEMAP_URLSET_OPEN,
        rows.join,
        "</urlset>"
      ].join
    end

    def join_url(*parts)
      joined = parts.compact.map(&:to_s).reject(&:empty?).join("/")
      "/#{joined}".gsub(%r{/+}, "/")
    end

    def format_date(value)
      return value.iso8601 if value.respond_to?(:iso8601)
      return value.to_s unless value.nil?

      nil
    end

    def sitemap_url_row(site, path, lastmod)
      location = self.class.xml_escape(self.class.absolute_url(site, self.class.pretty_url(path)))
      parts = ["<url><loc>", location, "</loc>"]

      lastmod_value = self.class.date_to_xmlschema(lastmod)
      if lastmod_value
        parts << "<lastmod>"
        parts << lastmod_value
        parts << "</lastmod>"
      end

      parts << "</url>"
      parts.join
    end

    class << self
      def absolute_url(site, input)
        return nil if input.nil?
        return input if Addressable::URI.parse(input.to_s).absolute?

        relative = relative_url(site, input)
        site_url = site.config["url"].to_s
        return relative if site_url.empty?

        Addressable::URI.parse(site_url + relative).normalize.to_s
      end

      def relative_url(site, input)
        return nil if input.nil?
        return input if Addressable::URI.parse(input.to_s).absolute?

        baseurl = site.config["baseurl"].to_s.chomp("/")
        path = ensure_leading_slash(input.to_s)

        Addressable::URI.parse("#{ensure_leading_slash(baseurl)}#{path}").normalize.to_s
      end

      def pretty_url(input)
        input.to_s.sub(%r!/index\.html$!, "/")
      end

      def ensure_leading_slash(input)
        return input if input.nil? || input.empty? || input.start_with?("/")

        "/#{input}"
      end

      def xml_escape(input)
        input.to_s.encode(:xml => :attr).gsub(%r!\A"|"\Z!, "")
      end

      def date_to_xmlschema(value)
        return nil if value.nil?

        date = Liquid::Utils.to_date(value)
        return nil unless date.respond_to?(:to_time)

        date.to_time.dup.localtime.xmlschema
      rescue ArgumentError, TypeError, Liquid::ArgumentError
        nil
      end

      def plain_text(content)
        return "" if content.nil? || content.empty?

        text = content.dup

        # Remove fence markers while keeping the code text itself searchable.
        text.gsub!(/^\s*```[^\n]*\n?/, "\n")
        text.gsub!(/^\s*~~~[^\n]*\n?/, "\n")

        text.gsub!(/!\[([^\]]*)\]\([^)]+\)/, '\1')
        text.gsub!(/\[([^\]]+)\]\([^)]+\)/, '\1')
        text.gsub!(/(?:^|\s)\{\:[^}]+\}/, " ")
        text.gsub!(/^\s*>\s*\[![^\]]+\][+-]?\s*/, "")
        text.gsub!(/^\s*>\s?/, "")
        text.gsub!(/^\s{0,3}[#]{1,6}\s+/, "")
        text.gsub!(/`([^`\n]+)`/, '\1')
        text.gsub!(/[*_~]/, "")
        text.gsub!(/<[^>]+>/m, " ")
        text.gsub!(/\\[\[\]\(\)]/, " ")
        text.gsub!(/^\s*[-*+]\s+/, "")
        text.gsub!(/^\s*\d+\.\s+/, "")

        text = CGI.unescapeHTML(text)
        text.tr!("\u00A0", " ")
        text.gsub!(/\s+/, " ")
        text.strip
      end

      def count_words(text)
        text.scan(/[[:alnum:]]+/).size
      end
    end
  end
end
