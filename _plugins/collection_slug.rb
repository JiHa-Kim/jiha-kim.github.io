# _plugins/collection_index_slug.rb
require "pathname"

Jekyll::Hooks.register :documents, :post_init do |doc|
  # Only collections (not posts/pages)
  next unless doc.collection && doc.collection.label

  # Only items named index.*
  next unless File.basename(doc.path, ".*") == "index"

  # Compute path relative to the collection root, which is robust to collections_dir
  coll_root = Pathname.new(doc.collection.directory) # e.g. <site>/collections/_crash-courses
  rel       = Pathname.new(doc.path).relative_path_from(coll_root)
  parts     = rel.each_filename.to_a                 # e.g. ["linear-algebra","1-foundations","index.md"]
  section_path = parts[0..-3].join("/")         # everything above the item folder
  item_slug    = parts[-2]
  doc.data["permalink"] ||= "/#{doc.collection.label}/#{section_path}/#{item_slug}/"
  # Need at least section/item/index.md
  next unless parts.size >= 3
  section_slug = parts[-3] # parent-of-parent
  item_slug    = parts[-2] # parent folder

  # Set slug if missing
  doc.data["slug"] ||= item_slug

  # Set pretty permalink if none given
  doc.data["permalink"] ||= "/#{doc.collection.label}/#{section_slug}/#{doc.data["slug"]}/"

  # Jekyll.logger.info("slugfix",
  #   "#{doc.relative_path} → #{doc.data["permalink"]} (slug=#{doc.data["slug"]})")
end
