/**
 * Interactive Widget Framework (IAW) Core Utilities
 * Shared logic for theme detection, canvas mapping, and UI consistency.
 */

window.IAW = (function() {
    const core = {};

    /**
     * Theme Manager
     * Synchronizes widget colors with the global `data-mode` (light/dark).
     */
    core.getThemeColors = function(rootElement) {
        const style = getComputedStyle(rootElement || document.documentElement);
        const isDark = document.documentElement.getAttribute('data-mode') === 'dark';

        return {
            // Identity Tokens (Calibrated for Matte Aesthetic)
            sourceA: isDark ? '#eab308' : '#d4a017', // Matte Yellow vs True Mustard
            sourceB: isDark ? '#0284c7' : '#0ea5e9', // Deep Sky vs Bright Sky
            target:  isDark ? '#0284c7' : '#0ea5e9', // Aqua for 1D mappings
            destination: isDark ? '#6366f1' : '#818cf8', // Indigo for Holes
            
            // Semantic Tokens
            success: style.getPropertyValue('--iaw-success').trim() || '#059669',
            danger:  style.getPropertyValue('--iaw-danger').trim() || '#dc2626',
            
            // Neutral Tokens
            text:    style.getPropertyValue('--iaw-text').trim(),
            muted:   style.getPropertyValue('--iaw-muted').trim(),
            border:  style.getPropertyValue('--iaw-border').trim(),
            bg:      style.getPropertyValue('--iaw-bg').trim(),
            surface: style.getPropertyValue('--iaw-surface').trim()
        };
    };

    /**
     * Utility: Mapping and Math
     */
    core.lerp = (a, b, t) => a + (b - a) * t;
    
    core.createMapper = function(domain, range) {
        const dSpan = domain.max - domain.min;
        const rSpan = range.max - range.min;
        return {
            toRange: (val) => range.min + ((val - domain.min) / dSpan) * rSpan,
            toDomain: (px) => domain.min + ((px - range.min) / rSpan) * dSpan
        };
    };

    /**
     * UI: Collision Resolution for overlapping labels or tags
     */
    core.resolveCollisions = function(nodes, options = {}) {
        if (!nodes || nodes.length <= 1) {
            nodes.forEach(n => {
                if (n.el) {
                    n.el.style.left = `${n.x}px`;
                    n.el.style.top = `${n.y}px`;
                }
            });
            return;
        }

        const gap = options.gap || 8;
        const processed = new Set();
        const clusters = [];

        // Build clusters of overlapping nodes
        for (let i = 0; i < nodes.length; i++) {
            if (processed.has(i)) continue;
            let cluster = [nodes[i]];
            processed.add(i);
            let changed = true;
            while (changed) {
                changed = false;
                for (let j = 0; j < nodes.length; j++) {
                    if (processed.has(j)) continue;
                    const n2 = nodes[j];
                    const overlaps = cluster.some(n1 => {
                        const dx = Math.abs(n1.x - n2.x);
                        const dy = Math.abs(n1.y - n2.y);
                        return dx < (n1.w + n2.w) / 2 + 10 && dy < (n1.h + n2.h) / 2 + 6;
                    });
                    if (overlaps) {
                        cluster.push(n2);
                        processed.add(j);
                        changed = true;
                    }
                }
            }
            clusters.push(cluster);
        }

        // Space out nodes within each cluster vertically
        clusters.forEach(cluster => {
            if (cluster.length <= 1) return;
            cluster.sort((a, b) => a.y - b.y);
            const totalH = cluster.reduce((sum, n) => sum + n.h, 0) + (cluster.length - 1) * gap;
            const targetY = cluster.reduce((sum, n) => sum + n.y, 0) / cluster.length;
            let curY = targetY - totalH / 2;
            cluster.forEach(n => {
                n.y = curY + n.h / 2;
                curY += n.h + gap;
            });
        });

        // Apply final positions
        nodes.forEach(n => {
            if (n.el) {
                n.el.style.left = `${n.x}px`;
                n.el.style.top = `${n.y}px`;
            }
        });
    };

    /**
     * Initialization Helper
     */
    core.observeTheme = function(callback) {
        const observer = new MutationObserver(callback);
        observer.observe(document.documentElement, { attributes: true, attributeFilter: ['data-mode'] });
        return observer;
    };

    return core;
})();
