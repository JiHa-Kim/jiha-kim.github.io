# frozen_string_literal: true

module Jekyll
  # This hook calculates related posts for all collections exactly once per generation.
  # It stores them in `doc.data['calculated_related_posts']` to avoid O(N^2 log N)
  # penalty when parsing via Liquid templates.
  Hooks.register :site, :post_read do |site|
    site.collections.each do |_, collection|
      docs = collection.docs
      
      # We only care about collections that have more than 1 document
      next unless docs.size > 1

      # Group by the specified sort key (default: path)
      # Liquid template checked for page.order, then page.date, else path.
      # We just default to date for posts, and path for everything else unless explicitly ordered.
      ordered_docs = docs.sort_by do |d|
        if d.data['order']
          [0, d.data['order']]
        elsif d.data['date']
          [1, d.data['date']]
        else
          [2, d.path]
        end
      end

      count = ordered_docs.size

      ordered_docs.each_with_index do |doc, idx|
        next_i = (idx + 1) % count
        prev_i = (idx - 1) % count

        # Deterministic random: seed from URL length
        seed = doc.url.nil? ? 0 : doc.url.length
        rand_i = (idx + seed) % count

        # Avoid duplicates (and self)
        if rand_i == idx || rand_i == next_i || rand_i == prev_i
          rand_i = (rand_i + 1) % count
        end
        rand_i = nil if count < 3

        picks = [next_i]
        picks << prev_i if prev_i != next_i
        picks << rand_i if rand_i

        related = picks.map { |i| ordered_docs[i] }.compact

        doc.data['calculated_related_posts'] = related
      end
    end
  end
end
