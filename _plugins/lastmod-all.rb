# _plugins/lastmod-only-post-fast.rb
require "open3"

PROD = ENV["JEKYLL_ENV"] == "production" || ENV["LASTMOD"] == "1"

def layout_is_post?(thing)
  layout = thing.data["layout"]
  layout && layout.to_s.split(".").first == "post"
end

def git_lastmod_iso(path)
  # single call; %cI = strict ISO-8601 of committer date
  out, _ = Open3.capture2(%(git --no-pager log -1 --pretty=%cI -- "#{path}"))
  ts = out.strip
  ts.empty? ? nil : ts
end

def maybe_set_lastmod!(thing)
  return unless PROD
  return unless layout_is_post?(thing)
  return if thing.data.key?("last_modified_at") # don't overwrite manual values
  return unless File.exist?(thing.path)         # skip generated/virtual

  if (ts = git_lastmod_iso(thing.path))
    # Optional: suppress "Updated" on first publish by matching the date part.
    # If you want the original Chirpy behavior (no "Updated" on first commit),
    # uncomment the next 4 lines to keep it hidden when dates match.
    #
    # if thing.data["date"]
    #   return if ts[0,10] == thing.data["date"].to_s[0,10]
    # end
    thing.data["last_modified_at"] = ts
  end
end

Jekyll::Hooks.register :documents, :post_init do |doc|
  maybe_set_lastmod!(doc)
end

Jekyll::Hooks.register :pages, :post_init do |page|
  maybe_set_lastmod!(page)
end
