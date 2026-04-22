# frozen_string_literal: true

require "digest"
require "fileutils"
require "json"
require "pathname"
require "jekyll"
require "jekyll-scholar"

module Jekyll
  module ScholarArtifactCache
    ARTIFACT_VERSION = "2026-04-22-scholar-reference-artifacts-v1".freeze
    ARTIFACT_RELATIVE_PATH = ".build_artifacts/scholar-references.json".freeze
    ARTIFACT_GEMS = %w[jekyll-scholar bibtex-ruby citeproc citeproc-ruby csl].freeze

    module_function

    ArtifactSite = Struct.new(:source, :config, :layouts, :site_payload)

    class ArtifactRenderer
      include Jekyll::Scholar::Utilities

      attr_reader :config, :context, :site

      def initialize(site:, bibliography:)
        @site = site
        @context = {}
        @config = Jekyll::Scholar.defaults.dup
        @config.merge!(site.config["scholar"] || {})
        @config["bibliography"] = bibliography
      end

      def index_sensitive?
        sample = entries.first
        return false unless sample

        converted = converted_entry(sample)
        csl_renderer(true)
        render_bibliography(converted, 1) != render_bibliography(converted, 2)
      ensure
        csl_renderer(true)
      end

      def bibliography_references
        csl_renderer(true)
        entries.each_with_index.each_with_object({}) do |(entry, index), references|
          references[entry.key.to_s] = render_bibliography(converted_entry(entry), index + 1)
        end
      end

      private

      def converted_entry(entry)
        return entry if bibtex_filters.empty?

        entry.dup.convert(*bibtex_filters)
      end
    end

    def disabled?
      ENV["JEKYLL_DISABLE_SCHOLAR_ARTIFACT_CACHE"] == "1"
    end

    def artifact_path(root)
      File.join(root, ARTIFACT_RELATIVE_PATH)
    end

    def artifact_manifest(site)
      return if disabled?

      if site.instance_variable_defined?(:@scholar_artifact_manifest)
        cached = site.instance_variable_get(:@scholar_artifact_manifest)
        return cached unless cached == false

        return nil
      end

      manifest = load_artifact_manifest(site)
      site.instance_variable_set(:@scholar_artifact_manifest, manifest || false)
      manifest
    end

    def load_artifact_manifest(site)
      path = artifact_path(site.source)
      return unless File.file?(path)

      parsed = JSON.parse(File.read(path))
      return parsed if valid_manifest?(parsed)

      Jekyll.logger.warn "ScholarArtifactCache:", "Ignoring unsupported artifact manifest at #{path}"
      nil
    rescue JSON::ParserError => e
      Jekyll.logger.warn "ScholarArtifactCache:", "Ignoring invalid artifact manifest at #{path}: #{e.message}"
      nil
    end

    def valid_manifest?(manifest)
      manifest.is_a?(Hash) &&
        manifest["version"] == ARTIFACT_VERSION &&
        manifest["contexts"].is_a?(Hash)
    end

    def artifact_reference_html(utilities, entry)
      return unless artifact_supported_for?(utilities, entry)

      manifest = artifact_manifest(utilities.site)
      return unless manifest

      context = manifest["contexts"][context_key_for_utilities(utilities)]
      return unless context
      return if context["index_sensitive"]

      context.dig("references", entry.key.to_s)
    end

    def artifact_supported_for?(utilities, entry)
      return false if disabled?
      return false unless entry
      return false unless utilities.reference_only_bibliography_template?
      return false if utilities.allow_locale_overrides?

      true
    end

    def context_key_for_utilities(utilities)
      metadata = context_metadata(
        site: utilities.site,
        style: utilities.style,
        locale: utilities.config["locale"],
        bibtex_filters: utilities.bibtex_filters,
        bibliography_paths: utilities.bibtex_paths
      )

      context_key(metadata)
    end

    def context_key(metadata)
      Digest::SHA1.hexdigest(JSON.generate(stable_value(metadata)))
    end

    def context_metadata(site:, style:, locale:, bibtex_filters:, bibliography_paths:)
      physical_paths = bibliography_paths.map { |path| File.expand_path(path, site.source) }

      {
        "artifact_version" => ARTIFACT_VERSION,
        "style" => style.to_s,
        "locale" => locale.to_s,
        "bibtex_filters" => Array(bibtex_filters).map(&:to_s),
        "bibliography_paths" => physical_paths.map { |path| site_relative_path(site, path) },
        "bibliography_digests" => physical_paths.each_with_object({}) do |path, digests|
          digests[site_relative_path(site, path)] = file_digest(site, path)
        end,
        "gem_versions" => gem_versions
      }
    end

    def gem_versions
      ARTIFACT_GEMS.each_with_object({}) do |name, versions|
        spec = Gem.loaded_specs[name]
        versions[name] = spec.version.to_s if spec
      end
    end

    def stable_value(value)
      case value
      when Hash
        value.keys.sort_by(&:to_s).each_with_object({}) do |key, result|
          result[key.to_s] = stable_value(value[key])
        end
      when Array
        value.map { |item| stable_value(item) }
      else
        value
      end
    end

    def site_relative_path(site, path)
      Pathname(File.expand_path(path)).relative_path_from(Pathname(site.source)).to_s
    end

    def file_digest(site, path)
      digests = site.instance_variable_get(:@scholar_artifact_file_digests) ||
        site.instance_variable_set(:@scholar_artifact_file_digests, {})

      digests[path] ||= Digest::SHA1.file(path).hexdigest
    rescue Errno::ENOENT
      ""
    end

    def generate_manifest(root:)
      site = build_artifact_site(root)
      contexts = {}

      discover_bibliographies(site).each do |bibliography|
        renderer = ArtifactRenderer.new(site: site, bibliography: bibliography)
        metadata = context_metadata(
          site: site,
          style: renderer.style,
          locale: renderer.config["locale"],
          bibtex_filters: renderer.bibtex_filters,
          bibliography_paths: renderer.bibtex_paths
        )
        context_id = context_key(metadata)
        next if contexts.key?(context_id)

        index_sensitive = renderer.index_sensitive?
        contexts[context_id] = metadata.merge(
          "bibliography" => bibliography,
          "entry_count" => renderer.entries.length,
          "index_sensitive" => index_sensitive,
          "references" => index_sensitive ? {} : renderer.bibliography_references
        )
      end

      {
        "version" => ARTIFACT_VERSION,
        "contexts" => contexts
      }
    end

    def build_artifact_site(root)
      config = Jekyll.configuration("source" => root)
      Jekyll::Cache.cache_dir ||= File.join(config["source"], ".jekyll-cache")
      ArtifactSite.new(config["source"], config, {}, {})
    end

    def discover_bibliographies(site)
      root = bibliography_root(site)
      return [] unless Dir.exist?(root)

      Dir.glob(File.join(root, "**/*.bib"))
        .select { |path| File.file?(path) }
        .sort
        .map { |path| Pathname(path).relative_path_from(Pathname(root)).to_s }
    end

    def bibliography_root(site)
      scholar_source = site.config.dig("scholar", "source") || Jekyll::Scholar.defaults["source"]
      candidate = File.expand_path(scholar_source, site.source)
      return candidate if Dir.exist?(candidate)

      scholar_source
    end

    def write_manifest!(root:)
      manifest = generate_manifest(root: root)
      path = artifact_path(root)
      FileUtils.mkdir_p(File.dirname(path))
      File.write(path, JSON.pretty_generate(manifest) + "\n")
      path
    end
  end
end
