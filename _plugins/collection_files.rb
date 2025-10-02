# # frozen_string_literal: true

# require "jekyll"
# require "pathname"

# module Jekyll
#   module CollectionFiles
#     class CollectionFile < StaticFile
#       # Initialize a new CollectionFile.
#       #
#       # site - The Site.
#       # base - The String path to the <source>
#       # dir  - The String path between <source> and the file
#       # name - The String filename of the file
#       # dest - The String path to the containing folder of the document which is output
#       def initialize(site, base, dir, name, dest)
#         super(site, base, dir, name)
#         @name = name
#         @dest = dest
#       end

#       # Obtain destination path.
#       #
#       # dest - The String path to the destination dir.
#       #
#       # Returns destination file path.
#       def destination(_dest)
#         File.join(@dest, @name)
#       end
#     end

#     class CollectionFileGenerator < Generator
#       def generate(site)
#         site_srcroot = Pathname.new(site.source)
#         all_docs_as_paths = site.documents.map { |doc| Pathname.new(doc.path) }

#         # Find all documents that are "post-like" i.e. have their own folder of assets
#         # This means their containing folder is not the collection's root folder.
#         docs_with_dirs = site.documents.reject do |doc|
#           collection_dir = Pathname.new(site.in_source_dir(doc.collection.directory))
#           Pathname.new(doc.path).dirname.eql?(collection_dir)
#         end

#         assets = docs_with_dirs.map do |doc|
#           dest_dir = Pathname.new(doc.destination("")).dirname
#           postdir = Pathname.new(doc.path).dirname

#           # Find all files under the post's directory
#           Dir[postdir + "**/*"].map do |fname|
#             asset_abspath = Pathname.new(fname)
#             # Reject directories and files that are documents themselves
#             next if File.directory?(asset_abspath)
#             next if all_docs_as_paths.include?(asset_abspath)

#             srcroot_to_asset = asset_abspath.relative_path_from(site_srcroot)
#             srcroot_to_assetdir = srcroot_to_asset.dirname
#             asset_basename = srcroot_to_asset.basename

#             assetdir_abs = site_srcroot + srcroot_to_assetdir
#             postdir_to_assetdir = assetdir_abs.relative_path_from(postdir)

#             CollectionFile.new(site, site_srcroot, srcroot_to_assetdir.to_path, asset_basename, (dest_dir + postdir_to_assetdir).to_path)
#           end.compact
#         end.flatten

#         site.static_files.concat(assets)
#       end
#     end
#   end
# end