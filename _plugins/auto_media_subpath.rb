# Auto-fills page.media_subpath for *any* document that:
#   1. lives in a collection (including posts), and
#   2. sits in its own directory (index.md pattern).

Jekyll::Hooks.register :documents, :pre_render do |doc|
    next unless doc.relative_path.end_with?('index.md')
  
    # slug = directory part between the collection root and index.md
    slug = File.dirname(doc.relative_path)
           .sub(%r!^[^/]+/!, '')          # strip "_posts/" or "_notes/" …
    collection_prefix = doc.collection.label
  
    # Only set it if the author hasn’t overridden it manually
    doc.data['media_subpath'] ||= "/#{collection_prefix}/#{slug}"
  end
  