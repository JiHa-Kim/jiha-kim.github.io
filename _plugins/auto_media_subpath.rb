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
      # ANY OTHER COLLECTION
      # Build a robust dir_slug that works whether the doc is:
      # - in nested folders (collections/_foo/dir1/dir2/index.md)
      # - a single file at collection root (collections/_foo/topic.md)
      # - a top-level index.md (collections/_foo/index.md)
      rel_path = doc.relative_path # e.g. "_series/test/1-test/1-test.md"
      parts = rel_path.split("/")

      # parts[0] is like "_series"
      # If it's nested, join everything between collection folder and filename.
      if parts.length >= 3
          dir_slug = parts[1..-2].join("/") # "test/1-test"
      else
          # No subdir: fall back to the filename (without ext), unless it's "index"
          base = doc.basename_without_ext
          dir_slug = (base.downcase == "index") ? "" : base
      end

      # Prepend the collection label (e.g., "series", "notes", "crash-courses")
      doc.data["media_subpath"] =
          if dir_slug.empty?
          "/#{doc.collection.label}"
          else
          "/#{doc.collection.label}/#{dir_slug}"
          end
      end
  end
