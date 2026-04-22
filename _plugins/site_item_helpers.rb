# frozen_string_literal: true

module Jekyll
  module SiteItemHelpers
    module_function

    def each_unique_content_item(site)
      return enum_for(__method__, site) unless block_given?

      seen = {}

      site.pages.each do |page|
        yield_unique(page, seen) { yield page }
      end

      site.collections.each_value do |collection|
        collection.docs.each do |doc|
          yield_unique(doc, seen) { yield doc }
        end
      end
    end

    def each_post_like_item(site)
      return enum_for(__method__, site) unless block_given?

      each_unique_content_item(site) do |item|
        next unless item.respond_to?(:data)
        next unless item.data["layout"] == "post"

        yield item
      end
    end

    def yield_unique(item, seen)
      key = if item.respond_to?(:relative_path) && item.relative_path
        item.relative_path
      elsif item.respond_to?(:path) && item.path
        item.path
      else
        item.object_id
      end

      return if seen[key]

      seen[key] = true
      yield
    end
    private_class_method :yield_unique
  end
end
