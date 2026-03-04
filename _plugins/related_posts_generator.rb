# frozen_string_literal: true

module Jekyll
  # This hook calculates related posts for all collections using a scoring
  # algorithm based on tags, categories, and series overlap.
  # Replaces the theme's O(N^2) Liquid implementation with an O(NK) Ruby one.
  Hooks.register :site, :post_read do |site|
    # 1. Gather all documents across all relevant collections
    all_docs = []
    site.collections.each do |label, collection|
      # Skip non-content collections like 'data'
      next if label == 'data'
      all_docs.concat(collection.docs)
    end

    # 2. Build reverse indices for fast lookup
    tag_to_docs = Hash.new { |h, k| h[k] = [] }
    cat_to_docs = Hash.new { |h, k| h[k] = [] }
    series_to_docs = Hash.new { |h, k| h[k] = [] }

    all_docs.each do |d|
      (d.data['tags'] || []).each { |t| tag_to_docs[t] << d }
      (d.data['categories'] || []).each { |c| cat_to_docs[c] << d }
      if d.data['series']
        series_to_docs[d.data['series']] << d
      end
    end

    # 3. For each document, identify its related counterparts
    all_docs.each do |doc|
      # Skip docs with no metadata to match on
      next if (doc.data['tags'] || []).empty? && 
              (doc.data['categories'] || []).empty? && 
              doc.data['series'].nil?

      scores = Hash.new(0)
      
      # Match tags (High weight: 1.0)
      (doc.data['tags'] || []).each do |t|
        tag_to_docs[t].each { |other| scores[other] += 1.0 }
      end
      
      # Match categories (Medium weight: 0.5)
      (doc.data['categories'] || []).each do |c|
        cat_to_docs[c].each { |other| scores[other] += 0.5 }
      end

      # Match series (High weight: 1.0)
      if doc.data['series']
        series_to_docs[doc.data['series']].each { |other| scores[other] += 1.0 }
      end

      # Remove current document from its own related list
      scores.delete(doc)

      # 4. Sort candidates by:
      #    - Total Score (Descending)
      #    - Date (Descending - Newer first)
      #    - Sort Index (Ascending - Series order)
      #    - Path (Ascending - Stability)
      sorted_related = scores.sort_by do |other, score|
        [
          -score,
          (other.data['date'] ? -other.data['date'].to_i : 0),
          (other.data['sort_index'] || 999), 
          other.path
        ]
      end

      # Pick the top 3 best-matching documents
      doc.data['calculated_related_posts'] = sorted_related.take(3).map(&:first)
    end
  end
end
