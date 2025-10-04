# _plugins/collection_index_slug.rb
require "pathname"

Jekyll::Hooks.register :documents, :post_init do |doc|
  # Only collection documents named index.*
  next unless doc.collection && doc.collection.label
  next unless File.basename(doc.path, ".*") == "index"

  coll_root = Pathname.new(doc.collection.directory)          # e.g. <site>/collections/_crash-courses
  rel       = Pathname.new(doc.path).relative_path_from(coll_root)
  parts     = rel.each_filename.to_a                          # ["linear-algebra","1-foundations","index.md"] etc.

  case parts.size
  when 1
    # collection root landing: _crash-courses/index.md
    doc.data["permalink"] ||= "/#{doc.collection.label}/"
    doc.data["layout"]    ||= "collection-landing"
    Jekyll.logger.info("slugfix", "#{doc.relative_path} → #{doc.data['permalink']} (root landing)")
  when 2
    # section landing: _crash-courses/<section>/index.md (also supports nested later via join)
    section_path = parts[0..-2].join("/")                     # "linear-algebra" (or "a/b" if nested)
    # optional: give it a slug if you like to use {{ page.slug }} in templates
    doc.data["slug"]      ||= File.basename(section_path)
    doc.data["permalink"] ||= "/#{doc.collection.label}/#{section_path}/"
    doc.data["layout"]    ||= "collection-landing"
    # Jekyll.logger.info("slugfix", "#{doc.relative_path} → #{doc.data['permalink']} (section landing)")
  else
    # item page: _crash-courses/<section_path>/<item>/index.md
    section_path = parts[0..-3].join("/")                      # "linear-algebra" or "a/b"
    item_slug    = parts[-2]
    doc.data["slug"]      ||= item_slug
    doc.data["permalink"] ||= "/#{doc.collection.label}/#{section_path}/#{doc.data['slug']}/"
    # Jekyll.logger.info("slugfix", "#{doc.relative_path} → #{doc.data['permalink']} (item slug=#{doc.data['slug']})")
  end
end
