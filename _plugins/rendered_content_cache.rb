# frozen_string_literal: true

require "digest"
require "jekyll"

module Jekyll
  module RenderedContentCache
    CACHE_NAMESPACE = "RenderedDocumentBody".freeze
    CACHE_VERSION = "2026-04-22-rendered-body-v1".freeze

    module_function

    def eligible?(renderer)
      document = renderer.document

      return false unless Jekyll.env == "production"
      return false if ENV["JEKYLL_DISABLE_RENDERED_CONTENT_CACHE"] == "1"
      return false unless document.respond_to?(:output_ext) && document.output_ext == ".html"
      return false if document.is_a?(Jekyll::Excerpt)
      return false if document.respond_to?(:asset_file?) && document.asset_file?
      return false if document.respond_to?(:yaml_file?) && document.yaml_file?
      return false if document.data["rendered_content_cache"] == false
      return false unless document.content && !document.content.empty?

      true
    end

    def cache(site)
      site.instance_variable_get(:@rendered_content_cache) ||
        site.instance_variable_set(:@rendered_content_cache, Jekyll::Cache.new(CACHE_NAMESPACE))
    end

    def cache_key(site, document)
      Digest::SHA1.hexdigest(Marshal.dump(cache_payload(site, document)))
    end

    def cache_payload(site, document)
      {
        "version" => CACHE_VERSION,
        "document" => {
          "path" => document.path,
          "url" => document.url,
          "layout" => document.data["layout"],
          "content_digest" => Digest::SHA1.hexdigest(document.content),
          "source_signature" => file_signature(document.path)
        },
        "bibliography_signature" => bibliography_signature(site, document),
        "site" => {
          "config" => relevant_config(site),
          "data_signature" => dependency_signature(site, "_data"),
          "includes_signature" => dependency_signature(site, "_includes"),
          "plugins_signature" => dependency_signature(site, "_plugins", pattern: "**/*.rb")
        }
      }
    end

    def relevant_config(site)
      site.instance_variable_get(:@rendered_content_cache_config_signature) ||
        site.instance_variable_set(
          :@rendered_content_cache_config_signature,
          {
            "url" => site.config["url"],
            "baseurl" => site.baseurl,
            "markdown" => site.config["markdown"],
            "kramdown" => site.config["kramdown"],
            "liquid" => site.config["liquid"],
            "scholar" => site.config["scholar"]
          }
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
      return render_document_without_body_cache unless RenderedContentCache.eligible?(self)

      info = render_document_info
      cache_key = RenderedContentCache.cache_key(site, document)
      cache = RenderedContentCache.cache(site)
      cached_body = cache.getset(cache_key) do
        render_document_body_for_cache(info)
      end

      document.content = cached_body.fetch("content").dup
      output = document.content.dup

      if document.place_in_layout?
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
  end
end
