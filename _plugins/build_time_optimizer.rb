# frozen_string_literal: true

require "cgi"
require "digest"
require "json"
require "jekyll"

module Jekyll
  class BuildTimeOptimizer < Generator
    priority :lowest

    CACHE_VERSION = "2026-04-22-plain-text-v1".freeze
    WORDS_PER_MINUTE = 180
    MIN_READ_TIME = 1
    MAX_TRENDING_TAGS = 10

    def generate(site)
      @cache ||= Jekyll::Cache.new("BuildTimeOptimizer")

      precompute_trending_tags(site)
      precompute_plain_text(site)
      precompute_search_index(site)
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
      each_candidate(site) do |item|
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

    def each_candidate(site)
      seen = {}

      site.posts.docs.each do |post|
        seen[post.path] = true
        yield post
      end

      site.pages.each do |page|
        next unless page.data["layout"] == "post"
        next if seen[page.path]

        seen[page.path] = true
        yield page
      end

      site.collections.each_value do |collection|
        collection.docs.each do |doc|
          next unless doc.data["layout"] == "post"
          next if seen[doc.path]

          seen[doc.path] = true
          yield doc
        end
      end
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

    class << self
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
