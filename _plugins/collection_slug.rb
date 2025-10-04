# _plugins/collection_index_slug.rb
require "pathname"

Jekyll::Hooks.register :documents, :post_init do |doc|
  # Only act on collection documents named index.*
  next unless doc.collection && doc.collection.label
  next unless File.basename(doc.path, ".*") == "index"

  # Path relative to the collection root (robust to collections_dir)
  coll_root = Pathname.new(doc.collection.directory) # e.g. <site>/collections/_crash-courses
  rel       = Pathname.new(doc.path).relative_path_from(coll_root)
  parts     = rel.each_filename.to_a                 # e.g. ["linear-algebra","1-foundations","index.md"]

  # Only rewrite true "items": <section>/<item>/index.md (size >= 3)
  # Skip collection root index.md (size == 1) and section landing index.md (size == 2)
  next unless parts.size >= 3

  # Everything above the item folder becomes the section path (supports nested sections)
  section_path = parts[0..-3].join("/")    # "linear-algebra" or "a/b" if nested
  item_slug    = parts[-2]                  # "1-foundations"

  # Set slug if missing and synthesize pretty permalink if none given
  doc.data["slug"]      ||= item_slug
  doc.data["permalink"] ||= "/#{doc.collection.label}/#{section_path}/#{doc.data["slug"]}/"

  # Jekyll.logger.info("slugfix",
  #   "#{doc.relative_path} → #{doc.data['permalink']} (slug=#{doc.data['slug']})")
end
