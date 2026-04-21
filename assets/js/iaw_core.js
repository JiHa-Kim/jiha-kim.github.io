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
    core.clamp = (value, min = 0, max = 1) => Math.max(min, Math.min(max, value));

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

    core.grayscaleColor = function(value) {
        const lightness = 10 + core.clamp(value) * 78;
        return `hsl(210, 18%, ${lightness.toFixed(1)}%)`;
    };

    core.maxAbs = function(matrix, floor = 1e-6) {
        return Math.max(floor, ...matrix.flat().map((value) => Math.abs(value)));
    };

    core.buildFrequencyExampleImage = function(preset, size = 8) {
        const image = [];

        for (let y = 0; y < size; y += 1) {
            const row = [];
            for (let x = 0; x < size; x += 1) {
                let value = 0;

                if (preset === 'smooth') {
                    const dx1 = x - 2.2;
                    const dy1 = y - 5.0;
                    const dx2 = x - 5.6;
                    const dy2 = y - 2.4;
                    value = 0.16
                        + 0.46 * Math.exp(-(dx1 * dx1 + dy1 * dy1) / 8.5)
                        + 0.26 * Math.exp(-(dx2 * dx2 + dy2 * dy2) / 11.0)
                        + 0.05 * (x / (size - 1));
                } else if (preset === 'edge') {
                    value = 0.16 + 0.66 / (1 + Math.exp(-(x - 3.5) * 1.7)) + 0.04 * (y / (size - 1) - 0.5);
                } else {
                    const checker = ((x + y) % 2 === 0) ? 1 : -1;
                    value = 0.5 + 0.23 * checker + 0.08 * Math.exp(-((x - 3.5) * (x - 3.5) + (y - 3.5) * (y - 3.5)) / 9);
                }

                row.push(core.clamp(value));
            }
            image.push(row);
        }

        return image;
    };

    core.drawMatrix = function(canvas, matrix, colorFn, options = {}) {
        if (!canvas || !matrix || !matrix.length || !matrix[0].length) return;

        const rect = canvas.getBoundingClientRect();
        if (rect.width < 1 || rect.height < 1) return;

        const dpr = window.devicePixelRatio || 1;
        canvas.width = rect.width * dpr;
        canvas.height = rect.height * dpr;

        const ctx = canvas.getContext('2d');
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, rect.width, rect.height);

        const rows = matrix.length;
        const cols = matrix[0].length;
        const cell = Math.max(1, Math.floor(Math.min(rect.width / cols, rect.height / rows)));
        const drawW = cell * cols;
        const drawH = cell * rows;
        const offsetX = Math.round((rect.width - drawW) / 2);
        const offsetY = Math.round((rect.height - drawH) / 2);

        for (let y = 0; y < rows; y += 1) {
            for (let x = 0; x < cols; x += 1) {
                ctx.fillStyle = colorFn(matrix[y][x], x, y);
                ctx.fillRect(offsetX + x * cell, offsetY + y * cell, Math.max(1, cell - 1), Math.max(1, cell - 1));
            }
        }

        if (options.borderColor) {
            ctx.strokeStyle = options.borderColor;
            ctx.lineWidth = options.borderWidth || 1;
            ctx.strokeRect(offsetX - 0.5, offsetY - 0.5, drawW, drawH);
        }
    };

    core.initFigure = function(rootOrId, options = {}) {
        const root = typeof rootOrId === 'string' ? document.getElementById(rootOrId) : rootOrId;
        if (!root || root.dataset.initialized === 'true') return null;
        if (options.requiresChart && !window.Chart) return null;
        root.dataset.initialized = 'true';
        return root;
    };

    core.mountFigure = function(rootOrId, initFn, options = {}) {
        const run = function() {
            const root = core.initFigure(rootOrId, options);
            if (!root) return;
            initFn(root);
        };

        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', run, { once: true });
        } else {
            run();
        }
    };

    core.bindSegmentedControl = function(rootOrButtons, callback, options = {}) {
        const buttons = Array.isArray(rootOrButtons) || NodeList.prototype.isPrototypeOf(rootOrButtons)
            ? Array.from(rootOrButtons)
            : Array.from(rootOrButtons.querySelectorAll(options.selector || '.iaw__seg-btn'));

        function activate(nextButton, notify = true) {
            if (!nextButton) return null;
            buttons.forEach((button) => button.classList.toggle('active', button === nextButton));
            if (notify && typeof callback === 'function') callback(nextButton);
            return nextButton;
        }

        buttons.forEach((button) => {
            button.addEventListener('click', () => activate(button, true));
        });

        if (options.triggerInitial) {
            activate(buttons.find((button) => button.classList.contains('active')) || buttons[0], true);
        }

        return {
            activate,
            buttons,
            getActive() {
                return buttons.find((button) => button.classList.contains('active')) || null;
            }
        };
    };

    core.requestMathTypeset = function(targets, attempt = 0) {
        if (window.MathJax && typeof window.MathJax.typesetPromise === 'function') {
            const list = Array.isArray(targets) ? targets.filter(Boolean) : [targets].filter(Boolean);
            if (list.length) {
                window.MathJax.typesetPromise(list).catch(() => {});
            }
            return;
        }

        if (attempt < 20) {
            window.setTimeout(() => core.requestMathTypeset(targets, attempt + 1), 250);
        }
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
