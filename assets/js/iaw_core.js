/**
 * Interactive Widget Framework (IAW) Core Utilities
 * Shared logic for theme detection, canvas mapping, and UI consistency.
 */

window.IAW = (function() {
    const core = {};
    const PLAY_ICON = '<svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>';
    const PAUSE_ICON = '<svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/></svg>';

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

    core.toAlpha = function(color, alpha) {
        const trimmed = String(color || '').trim();
        const hexMatch = trimmed.match(/^#([0-9a-f]{3}|[0-9a-f]{6})$/i);

        if (hexMatch) {
            let hex = hexMatch[1];
            if (hex.length === 3) {
                hex = hex.split('').map((char) => char + char).join('');
            }

            const r = parseInt(hex.slice(0, 2), 16);
            const g = parseInt(hex.slice(2, 4), 16);
            const b = parseInt(hex.slice(4, 6), 16);
            return `rgba(${r}, ${g}, ${b}, ${alpha})`;
        }

        const rgbMatch = trimmed.match(/^rgba?\(([^)]+)\)$/i);
        if (rgbMatch) {
            const parts = rgbMatch[1].split(',').map((part) => part.trim());
            return `rgba(${parts[0]}, ${parts[1]}, ${parts[2]}, ${alpha})`;
        }

        return trimmed;
    };
    
    core.createMapper = function(domain, range) {
        const dSpan = domain.max - domain.min;
        const rSpan = range.max - range.min;
        return {
            toRange: (val) => range.min + ((val - domain.min) / dSpan) * rSpan,
            toDomain: (px) => domain.min + ((px - range.min) / rSpan) * dSpan
        };
    };

    core.initFigure = function(rootOrId, options = {}) {
        const root = typeof rootOrId === 'string' ? document.getElementById(rootOrId) : rootOrId;
        if (!root || root.dataset.initialized === 'true') return null;
        if (options.requiresChart && !window.Chart) return null;
        root.dataset.initialized = 'true';
        return root;
    };

    core.setPlayButtonState = function(button, isPlaying) {
        if (!button) return;
        button.innerHTML = isPlaying ? PAUSE_ICON : PLAY_ICON;
        button.setAttribute('aria-pressed', isPlaying ? 'true' : 'false');
    };

    core.createPlaybackController = function(options = {}) {
        const minFrame = Number(options.minFrame ?? 0);
        const maxFrame = Number(options.maxFrame ?? 100);
        const frameDuration = Number(options.frameDuration ?? 20);
        const button = options.button || null;
        const slider = options.slider || null;
        const onFrameChange = typeof options.onFrameChange === 'function' ? options.onFrameChange : () => {};
        const onPlayStateChange = typeof options.onPlayStateChange === 'function' ? options.onPlayStateChange : () => {};

        let currentFrame = Number(options.initialFrame ?? (slider ? slider.value : minFrame));
        if (!Number.isFinite(currentFrame)) currentFrame = minFrame;
        currentFrame = Math.max(minFrame, Math.min(maxFrame, currentFrame));

        let isPlaying = false;
        let animationFrameId = null;
        let lastTimestamp = 0;

        function syncFrame(notify = true) {
            if (slider && Number(slider.value) !== currentFrame) {
                slider.value = String(currentFrame);
            }
            if (notify) onFrameChange(currentFrame);
        }

        function setPlayState(nextValue) {
            isPlaying = nextValue;
            core.setPlayButtonState(button, isPlaying);
            onPlayStateChange(isPlaying);
        }

        function stop() {
            if (animationFrameId) {
                cancelAnimationFrame(animationFrameId);
                animationFrameId = null;
            }
            lastTimestamp = 0;
            if (isPlaying) {
                setPlayState(false);
            } else {
                core.setPlayButtonState(button, false);
            }
        }

        function step(timestamp) {
            if (!isPlaying) return;

            if (!lastTimestamp) {
                lastTimestamp = timestamp;
            }

            if (timestamp - lastTimestamp >= frameDuration) {
                lastTimestamp = timestamp;

                if (currentFrame >= maxFrame) {
                    stop();
                    return;
                }

                currentFrame += 1;
                syncFrame(true);
            }

            animationFrameId = requestAnimationFrame(step);
        }

        function play() {
            if (isPlaying) return;
            if (currentFrame >= maxFrame) {
                currentFrame = minFrame;
                syncFrame(true);
            }
            setPlayState(true);
            animationFrameId = requestAnimationFrame(step);
        }

        function toggle() {
            if (isPlaying) stop();
            else play();
        }

        function setFrame(nextFrame, config = {}) {
            if (config.stopPlayback !== false) stop();
            currentFrame = Math.max(minFrame, Math.min(maxFrame, Number(nextFrame)));
            if (!Number.isFinite(currentFrame)) currentFrame = minFrame;
            if (config.notify === false) {
                if (slider && Number(slider.value) !== currentFrame) {
                    slider.value = String(currentFrame);
                }
            } else {
                syncFrame(true);
            }
            return currentFrame;
        }

        const handleButtonClick = () => toggle();
        const handleSliderInput = (event) => {
            setFrame(event.target.value, { stopPlayback: true, notify: true });
        };

        if (button) button.addEventListener('click', handleButtonClick);
        if (slider) slider.addEventListener('input', handleSliderInput);

        core.setPlayButtonState(button, false);
        if (options.syncInitial !== false && slider && Number(slider.value) !== currentFrame) {
            slider.value = String(currentFrame);
        }

        return {
            destroy() {
                stop();
                if (button) button.removeEventListener('click', handleButtonClick);
                if (slider) slider.removeEventListener('input', handleSliderInput);
            },
            getFrame() {
                return currentFrame;
            },
            isPlaying() {
                return isPlaying;
            },
            play,
            setFrame,
            stop,
            toggle
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
