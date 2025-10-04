# _plugins/lastmod-only-post-layout.rb
require "open3"

def in_git_repo?
  system("git rev-parse --is-inside-work-tree >NUL 2>&1") ||
    system("git rev-parse --is-inside-work-tree >/dev/null 2>&1")
end

def layout_is_post?(thing)
  # robust against "post.html" etc.
  layout = thing.data["layout"]
  layout && layout.to_s.split(".").first == "post"
end

def set_lastmod_from_git!(thing)
  return unless in_git_repo?
  return unless layout_is_post?(thing)
  return if thing.data.key?("last_modified_at") # don't overwrite manual values

  out, _ = Open3.capture2(%(git rev-list --count HEAD -- "#{thing.path}"))
  return unless out.to_i > 1

  ts, _ = Open3.capture2(%(git log -1 --pretty="%ad" --date=iso-strict -- "#{thing.path}"))
  ts = ts.strip
  thing.data["last_modified_at"] = ts unless ts.empty?
end

# Run after objects are initialized so front-matter defaults (incl. layout) are available.
Jekyll::Hooks.register :documents, :post_init do |doc|
  set_lastmod_from_git!(doc)
end

Jekyll::Hooks.register :pages, :post_init do |page|
  set_lastmod_from_git!(page)
end
