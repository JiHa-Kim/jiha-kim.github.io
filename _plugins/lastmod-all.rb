# _plugins/lastmod-all.rb
require "open3"
require "fileutils"
require "pathname"

PROD = ENV["JEKYLL_ENV"] == "production" || ENV["LASTMOD"] == "1"

module LastmodCache
  class << self
    def get_cache(site_source)
      return @cache if @cache
      @cache = {}
      return @cache unless PROD

      # Populating cache with a single git command
      # This is MUCH faster than spawning git for every file
      Dir.chdir(site_source) do
        Open3.popen3('git log --format="%cI" --name-only') do |_stdin, stdout, _stderr, _wait_thr|
          current_date = nil
          stdout.each_line do |line|
            line = line.strip
            if line =~ /^\d{4}-\d{2}-\d{2}T/
              current_date = line
            elsif !line.empty? && current_date
              # Only set if not already set (latest commit first)
              @cache[line] ||= current_date
            end
          end
        end
      end
      @cache
    end
  end
end

def layout_is_post?(thing)
  layout = thing.data["layout"]
  layout && layout.to_s.split(".").first == "post"
end

def maybe_set_lastmod!(thing)
  return unless PROD
  return unless layout_is_post?(thing)
  return if thing.data.key?("last_modified_at")
  return unless File.exist?(thing.path)

  # Normalize path relative to site source
  site_source = File.expand_path(thing.site.source)
  thing_path = File.expand_path(thing.path)
  rel_path = Pathname.new(thing_path).relative_path_from(Pathname.new(site_source)).to_s

  cache = LastmodCache.get_cache(site_source)
  if (ts = cache[rel_path])
    thing.data["last_modified_at"] = ts
  end
end

Jekyll::Hooks.register :documents, :post_init do |doc|
  maybe_set_lastmod!(doc)
end

Jekyll::Hooks.register :pages, :post_init do |page|
  maybe_set_lastmod!(page)
end
