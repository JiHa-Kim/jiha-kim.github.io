# _plugins/fix_post_asset_urls.rb
# This plugin post-processes HTML output for posts to ensure that asset URLs
# are relative to the post’s directory.
# It looks for src and href attributes whose value starts with "/" or "/./"
# and, if the URL isn’t already prefixed with the post’s URL, it prepends it.
#
# For example, if a post’s URL is "/posts/test/" and its HTML contains:
#   <img src="/gbm_path.png">
# then it will be rewritten to:
#   <img src="/posts/test/gbm_path.png">
#
# Be careful not to affect truly external links (which usually start with http:// or https://).
#
# Adjust the logic below if you need to fine‑tune which URLs to rewrite.

Jekyll::Hooks.register :documents, :post_render do |doc|
  # Only process HTML output for posts (adjust if you also want to process pages)
  if doc.output_ext == ".html" && doc.collection.label == "posts"
    post_url = doc.url # e.g., "/posts/test/"
    # Ensure post_url ends with a slash
    post_url += "/" unless post_url.end_with?("/")

    # Use a regex to find src and href attributes that start with "/" or "/./"
    # We ignore links that start with "http" or "https"
    doc.output.gsub!(%r{(href|src)="(/\.?/)([^"]+)"}) do
      attr = Regexp.last_match(1)
      prefix = Regexp.last_match(2)  # could be "/" or "/./"
      path   = Regexp.last_match(3)
      full_url = prefix + path

      # Skip rewriting if full_url already looks absolute with http(s)
      if full_url =~ %r{^https?://}
        %(#{attr}="#{full_url}")
      # If the full URL already begins with the post URL, leave it unchanged.
      elsif full_url.start_with?(post_url)
        %(#{attr}="#{full_url}")
      else
        # Otherwise, assume the intended relative path is just the filename.
        # Construct a new URL by joining the post_url with the path.
        # File.join takes care of removing extra slashes.
        new_url = File.join(post_url, path)
        %(#{attr}="#{new_url}")
      end
    end
  end
end
