// Handwriting animation — filled glyph revealed left-to-right via SVG clipPath

var svgCanvas = document.getElementById("canvas");
var svgCursor = document.getElementById("svg-cursor");
var loadingEl = document.getElementById("loading");

var font = null;
var INK_COLOR = "#1a1a2e";
var FONT_SIZE = 100;
var LINE_HEIGHT = 190;
var LEFT_MARGIN = 80;
var DRAW_DURATION = 400;

var cursorX = LEFT_MARGIN;
var cursorY = LINE_HEIGHT;

var lines = [[]];
var currentLineIdx = 0;
var clipIdCounter = 0;
var lastEnterTime = 0;
var lastKeyTime = {};
var KEY_DEBOUNCE = 800;

var svgNS = "http://www.w3.org/2000/svg";

// Shared <defs> for clipPaths
var defs = document.createElementNS(svgNS, "defs");
svgCanvas.insertBefore(defs, svgCanvas.firstChild);

// ============ Cursor ============
var updateCursor = function() {
    var rect = svgCursor.querySelector("rect");
    rect.setAttribute("x", cursorX);
    rect.setAttribute("y", cursorY - 30);
};

// ============ Add Letter ============
var addLetter = function(ch) {
    if (!font) return;

    var glyph = font.charToGlyph(ch);
    var scale = FONT_SIZE / font.unitsPerEm;
    var advance = (glyph.advanceWidth || font.unitsPerEm * 0.3) * scale;

    // Kerning with previous character
    var lineLetters = lines[currentLineIdx];
    if (lineLetters.length > 0) {
        var prevCh = lineLetters[lineLetters.length - 1].ch;
        var prevGlyph = font.charToGlyph(prevCh);
        var kern = font.getKerningValue(prevGlyph, glyph);
        cursorX += kern * scale;
    }

    // Get SVG path data
    var path = font.getPath(ch, cursorX, cursorY, FONT_SIZE);
    var pathData = path.toPathData(2);

    // Space or non-rendering character
    if (!pathData || pathData.trim() === "" || ch === " ") {
        cursorX += advance;
        lineLetters.push({ group: null, advance: advance, ch: ch, clipEl: null });
        updateCursor();
        return;
    }

    // Get bounding box of the glyph path for the clip rect
    var bbox = path.getBoundingBox();
    var pad = 20; // padding so flourishes aren't cut

    // Create a unique clipPath with a rect that animates from zero-width to full-width
    var clipId = "clip-" + (clipIdCounter++);
    var clipPathEl = document.createElementNS(svgNS, "clipPath");
    clipPathEl.setAttribute("id", clipId);

    var clipRect = document.createElementNS(svgNS, "rect");
    clipRect.setAttribute("x", bbox.x1 - pad);
    clipRect.setAttribute("y", bbox.y1 - pad);
    clipRect.setAttribute("width", "0");
    clipRect.setAttribute("height", bbox.y2 - bbox.y1 + pad * 2);
    clipPathEl.appendChild(clipRect);
    defs.appendChild(clipPathEl);

    // Create the filled letter path, clipped by the animated rect
    var g = document.createElementNS(svgNS, "g");
    g.setAttribute("clip-path", "url(#" + clipId + ")");

    var fillPath = document.createElementNS(svgNS, "path");
    fillPath.setAttribute("d", pathData);
    fillPath.setAttribute("fill", INK_COLOR);
    fillPath.setAttribute("stroke", "none");
    g.appendChild(fillPath);

    svgCanvas.insertBefore(g, svgCursor);

    // Animate the clip rect width from 0 to full width
    var fullWidth = bbox.x2 - bbox.x1 + pad * 2;

    // Use Web Animations API for smooth animation
    clipRect.animate([
        { width: "0px" },
        { width: fullWidth + "px" }
    ], {
        duration: DRAW_DURATION,
        easing: "ease-out",
        fill: "forwards"
    });

    // After animation, remove clip so glyph is fully visible (no clipping at all)
    setTimeout(function() {
        g.removeAttribute("clip-path");
        clipPathEl.remove();
    }, DRAW_DURATION + 50);

    cursorX += advance;
    lineLetters.push({ group: g, advance: advance, ch: ch, clipEl: clipPathEl });
    updateCursor();
};

// ============ Remove Letter ============
var removeLetter = function() {
    var lineLetters = lines[currentLineIdx];

    if (lineLetters.length > 0) {
        var entry = lineLetters.pop();
        if (entry.group) entry.group.remove();
        if (entry.clipEl) entry.clipEl.remove();
        cursorX -= entry.advance;
        updateCursor();
    } else if (currentLineIdx > 0) {
        currentLineIdx--;
        lines.pop();

        var prevLine = lines[currentLineIdx];
        cursorX = LEFT_MARGIN;
        for (var i = 0; i < prevLine.length; i++) {
            cursorX += prevLine[i].advance;
        }
        cursorY -= LINE_HEIGHT;
        updateCursor();
    }
};

// ============ New Line ============
var newLine = function() {
    currentLineIdx++;
    lines.push([]);
    cursorX = LEFT_MARGIN;
    cursorY += LINE_HEIGHT;
    updateCursor();

    var svgHeight = parseInt(svgCanvas.getAttribute("height")) || window.innerHeight;
    if (cursorY + LINE_HEIGHT > svgHeight) {
        svgCanvas.setAttribute("height", cursorY + LINE_HEIGHT * 2);
    }

    svgCursor.scrollIntoView({ behavior: "smooth", block: "nearest" });
};

// ============ Clear All ============
var clearAll = function() {
    var groups = svgCanvas.querySelectorAll("g:not(#svg-cursor)");
    groups.forEach(function(g) { g.remove(); });

    // Clear all clipPaths from defs
    while (defs.firstChild) defs.removeChild(defs.firstChild);

    lines = [[]];
    currentLineIdx = 0;
    cursorX = LEFT_MARGIN;
    cursorY = LINE_HEIGHT;
    updateCursor();
};

// ============ Keyboard Handler ============
document.addEventListener("keydown", function(e) {
    if (!font) return;
    if (e.metaKey || e.ctrlKey || e.altKey) return;

    if (e.key === "Backspace") {
        e.preventDefault();
        removeLetter();
    } else if (e.key === "Enter") {
        e.preventDefault();
        var now = Date.now();
        if (now - lastEnterTime < 500) return;
        lastEnterTime = now;
        newLine();
    } else if (e.key === "Escape") {
        e.preventDefault();
        clearAll();
    } else if (e.key.length === 1) {
        e.preventDefault();
        var now2 = Date.now();
        if (lastKeyTime[e.key] && now2 - lastKeyTime[e.key] < KEY_DEBOUNCE) return;
        lastKeyTime[e.key] = now2;
        addLetter(e.key);
    }
});

document.addEventListener("click", function() {
    if (svgCursor) svgCursor.scrollIntoView({ behavior: "smooth", block: "nearest" });
});

// ============ Load Font ============
opentype.load("priestacy/Priestacy.otf", function(err, f) {
    if (err) {
        loadingEl.textContent = "Error loading font: " + err;
        return;
    }
    font = f;
    loadingEl.remove();
    updateCursor();
});
