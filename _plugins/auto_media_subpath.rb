#
# auto_media_subpath.rb
#
# Purpose:
#   1) For posts (collection "posts") whose source is e.g.
#        _posts/2025-04-28-my-post/2025-04-28-my-post.md
#      this hook will strip off "YYYY-MM-DD-" from the filename to get the slug
#      ("my-post") and set
#        page.media_subpath = "/posts/my-post"
#      which exactly matches the final output folder under /posts/my-post/.
#
#   2) For any other collection (e.g. _notes, _series, _crash-courses, etc.),
#      it looks at the directory name inside that collection to form:
#        page.media_subpath = "/<collection-label>/<that-directory>"
#      so that if you have
#        collections/_notes/martingales/index.md
#      it becomes
#        page.media_subpath = "/notes/martingales"
#
#   You can still override per-page by adding `media_subpath: …` in front-matter.

Jekyll::Hooks.register :documents, :pre_render do |doc|
    # If the user already set media_subpath manually, do nothing.
    next if doc.data.key?('media_subpath')
  
    case doc.collection.label
    when 'posts'
      #
      # POSTS: source looks like
      #   _posts/2025-04-28-some-slug/2025-04-28-some-slug.md
      # We want to end up with
      #   media_subpath: "/posts/some-slug"
      #
      # 1) Get the base filename without extension, e.g.:
      #      "2025-04-28-some-slug"
      raw = doc.basename_without_ext
  
      # 2) Strip off the leading "YYYY-MM-DD-"
      slug = raw.sub(/^\d{4}-\d{2}-\d{2}-/, '')
  
      # 3) Assign exactly the same path as the final URL folder:
      doc.data['media_subpath'] = "/posts/#{slug}"
  
    else
      #
      # ANY OTHER COLLECTION: (_notes, _series, _crash-courses, etc.)
      # Suppose the source is
      #   collections/_notes/martingales/index.md
      # Then:
      #   doc.relative_path => "_notes/martingales/index.md"
      #   File.dirname(...) => "_notes/martingales"
      # We strip "_notes/" to get "martingales",
      # then join with "/notes" to form "/notes/martingales".
      #
      rel_dir = File.dirname(doc.relative_path)
      # If the file is not inside a subdirectory, skip
      # (it might be a top-level collection page).
      next if rel_dir.nil? || rel_dir == '.' || rel_dir.empty?
  
      # Remove the leading "<_collection_name>/" segment,
      # e.g. "_notes/martingales" → "martingales"
      dir_slug = rel_dir.sub(%r!^[^/]+/!, '')
  
      # Prepend the collection label (which is "notes", "series", "crash-courses", …)
      doc.data['media_subpath'] = "/#{doc.collection.label}/#{dir_slug}"
    end
  end
  