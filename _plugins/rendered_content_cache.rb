# frozen_string_literal: true

require "digest"
require "jekyll"
require "set"
require_relative "site_item_helpers"

module Jekyll
  module RenderedContentCache
    BODY_CACHE_NAMESPACE = "RenderedDocumentBody".freeze
    BODY_CACHE_VERSION = "2026-04-22-rendered-body-v1".freeze
    LAYOUT_CACHE_NAMESPACE = "RenderedLayoutOutput".freeze
    LAYOUT_CACHE_VERSION = "2026-04-22-rendered-layout-v1".freeze

    module_function

    def cache_disabled?
      ENV["JEKYLL_DISABLE_RENDERED_CONTENT_CACHE"] == "1"
    end

    def body_cache_eligible?(renderer)
      render_cache_eligible?(renderer) && renderer.document.content && !renderer.document.content.empty?
    end

    def layout_cache_eligible?(renderer)
      render_cache_eligible?(renderer) && renderer.document.place_in_layout?
    end

    def render_cache_eligible?(renderer)
      document = renderer.document

      return false unless Jekyll.env == "production"
      return false if cache_disabled?
      return false unless document.respond_to?(:output_ext) && document.output_ext == ".html"
      return false if document.is_a?(Jekyll::Excerpt)
      return false if document.respond_to?(:asset_file?) && document.asset_file?
      return false if document.respond_to?(:yaml_file?) && document.yaml_file?
      return false if document.data["rendered_content_cache"] == false

      true
    end

    def body_cache(site)
      site.instance_variable_get(:@rendered_document_body_cache) ||
        site.instance_variable_set(:@rendered_document_body_cache, Jekyll::Cache.new(BODY_CACHE_NAMESPACE))
    end

    def layout_cache(site)
      site.instance_variable_get(:@rendered_layout_output_cache) ||
        site.instance_variable_set(:@rendered_layout_output_cache, Jekyll::Cache.new(LAYOUT_CACHE_NAMESPACE))
    end

    def body_cache_key(site, document)
      Digest::SHA1.hexdigest(Marshal.dump(body_cache_payload(site, document)))
    end

    def layout_cache_key(site, document, body_content)
      Digest::SHA1.hexdigest(Marshal.dump(layout_cache_payload(site, document, body_content)))
    end

    def body_cache_payload(site, document)
      {
        "version" => BODY_CACHE_VERSION,
        "document" => {
          "path" => document.path,
          "url" => document.url,
          "layout" => document.data["layout"],
          "content_digest" => Digest::SHA1.hexdigest(document.content),
          "source_signature" => file_signature(document.path)
        },
        "bibliography_signature" => bibliography_signature(site, document),
        "site" => {
          "config" => config_signature(site),
          "data_signature" => dependency_signature(site, "_data"),
          "includes_signature" => dependency_signature(site, "_includes"),
          "plugins_signature" => dependency_signature(site, "_plugins", pattern: "**/*.rb"),
          "site_items_signature" => site_items_signature(site)
        }
      }
    end

    def layout_cache_payload(site, document, body_content)
      {
        "version" => LAYOUT_CACHE_VERSION,
        "document" => {
          "path" => document.path,
          "url" => document.url,
          "layout" => document.data["layout"],
          "data_digest" => payload_digest(document.data.to_h),
          "body_digest" => Digest::SHA1.hexdigest(body_content.to_s)
        },
        "bibliography_signature" => bibliography_signature(site, document),
        "site" => {
          "config" => config_signature(site),
          "data_signature" => dependency_signature(site, "_data"),
          "includes_signature" => dependency_signature(site, "_includes"),
          "layouts_signature" => layouts_signature(site),
          "plugins_signature" => dependency_signature(site, "_plugins", pattern: "**/*.rb"),
          "site_items_signature" => site_items_signature(site)
        }
      }
    end

    def config_signature(site)
      site.instance_variable_get(:@rendered_content_cache_config_signature) ||
        site.instance_variable_set(
          :@rendered_content_cache_config_signature,
          Digest::SHA1.hexdigest(site.config.inspect)
        )
    end

    def dependency_signature(site, relative_dir, pattern: "**/*")
      signatures = site.instance_variable_get(:@rendered_content_cache_dependency_signatures) ||
        site.instance_variable_set(:@rendered_content_cache_dependency_signatures, {})
      cache_key = [relative_dir, pattern]

      signatures[cache_key] ||= begin
        root = File.join(site.source, relative_dir)
        unless Dir.exist?(root)
          ""
        else
          entries = Dir.glob(File.join(root, pattern)).select { |path| File.file?(path) }
          digest_input = entries.sort.map do |path|
            relative_path = path.delete_prefix("#{site.source}/")
            "#{relative_path}:#{file_signature(path)}"
          end

          Digest::SHA1.hexdigest(digest_input.join("\n"))
        end
      end
    end

    def layouts_signature(site)
      dependency_signature(site, "_layouts")
    end

    def bibliography_signature(site, document)
      bibliography_entries = [
        site.config.dig("scholar", "bibliography"),
        document.data.dig("scholar", "bibliography")
      ].compact.uniq

      paths = bibliography_entries.flat_map do |entry|
        if entry.include?("*")
          Dir.glob(File.join(site.source, entry)).select { |path| File.file?(path) }
        else
          path = entry.start_with?("/") ? entry : File.join(site.source, entry)
          File.file?(path) ? [path] : []
        end
      end

      Digest::SHA1.hexdigest(
        paths.sort.map { |path| "#{path}:#{file_signature(path)}" }.join("\n")
      )
    end

    def payload_digest(value)
      Digest::SHA1.hexdigest(Marshal.dump(stable_cache_value(value)))
    end

    def stable_cache_value(value)
      case value
      when nil, true, false, Numeric, String
        value
      when Symbol
        value.to_s
      when Time, Date, DateTime
        value.iso8601(6)
      when Array
        value.map { |item| stable_cache_value(item) }
      when Hash
        value.keys.sort_by(&:to_s).each_with_object({}) do |key, result|
          result[key.to_s] = stable_cache_value(value[key])
        end
      else
        if value.respond_to?(:relative_path) || value.respond_to?(:path) || value.respond_to?(:url)
          {
            "__type__" => value.class.name,
            "relative_path" => value.respond_to?(:relative_path) ? value.relative_path : nil,
            "path" => value.respond_to?(:path) ? value.path : nil,
            "url" => value.respond_to?(:url) ? value.url : nil
          }.compact
        elsif value.respond_to?(:to_liquid) && value.method(:to_liquid).owner != Object
          stable_cache_value(value.to_liquid)
        else
          value.to_s
        end
      end
    end

    def site_items_signature(site)
      site.instance_variable_get(:@rendered_content_cache_site_items_signature) ||
        site.instance_variable_set(
          :@rendered_content_cache_site_items_signature,
          begin
            digest_input = []

            Jekyll::SiteItemHelpers.each_unique_content_item(site) do |item|
              path = item.respond_to?(:path) ? item.path : item.to_s
              relative_path = if path.start_with?("#{site.source}/")
                path.delete_prefix("#{site.source}/")
              else
                path
              end
              digest_input << "#{relative_path}:#{file_signature(path)}"
            end

            Digest::SHA1.hexdigest(digest_input.sort.join("\n"))
          end
        )
    end

    def file_signature(path)
      stat = File.stat(path)
      "#{stat.size}:#{stat.mtime.to_f}"
    rescue Errno::ENOENT
      ""
    end
  end

  class Renderer
    alias_method :render_document_without_body_cache, :render_document

    def render_document
      return render_document_without_body_cache unless RenderedContentCache.render_cache_eligible?(self)

      info = render_document_info
      cached_body = if RenderedContentCache.body_cache_eligible?(self)
        body_cache_key = RenderedContentCache.body_cache_key(site, document)
        RenderedContentCache.body_cache(site).getset(body_cache_key) do
          render_document_body_for_cache(info)
        end
      else
        render_document_body_for_cache(info)
      end

      document.content = cached_body.fetch("content").dup
      output = document.content.dup

      if RenderedContentCache.layout_cache_eligible?(self)
        output = render_layouts_with_cache(output, info)
      elsif document.place_in_layout?
        Jekyll.logger.debug "Rendering Layout:", document.relative_path
        output = place_in_layouts(output, payload, info)
      end

      output
    end

    private

    def render_document_info
      {
        :registers => { :site => site, :page => payload["page"] },
        :strict_filters => liquid_options["strict_filters"],
        :strict_variables => liquid_options["strict_variables"]
      }
    end

    def render_document_body_for_cache(info)
      output = document.content

      if document.render_with_liquid?
        Jekyll.logger.debug "Rendering Liquid:", document.relative_path
        output = render_liquid(output, payload, info, document.path)
      end

      Jekyll.logger.debug "Rendering Markup:", document.relative_path
      output = convert(output.to_s)
      document.content = output

      Jekyll.logger.debug "Post-Convert Hooks:", document.relative_path
      document.trigger_hooks(:post_convert)

      { "content" => document.content.dup }
    end

    def render_layouts_with_cache(output, info)
      layout_cache_key = RenderedContentCache.layout_cache_key(site, document, output)
      cached_layout = RenderedContentCache.layout_cache(site).getset(layout_cache_key) do
        {
          "content" => document.content.dup,
          "output" => place_in_layouts(output, payload, info)
        }
      end

      document.content = cached_layout.fetch("content").dup
      register_layout_dependencies
      cached_layout.fetch("output").dup
    end

    def register_layout_dependencies
      layout = layouts[document.data["layout"].to_s]
      validate_layout(layout)
      return unless layout

      used = Set.new([layout])

      loop do
        add_regenerator_dependencies(layout)
        layout = site.layouts[layout.data["layout"]]
        break unless layout
        break if used.include?(layout)

        used << layout
      end
    end
  end
end
