using System.Collections.Generic;
using Unity.Collections;
using UnityEngine;

namespace AnimationTools.Editor
{
/// <summary>
/// Writes a short label into the top-left of a readback image with a built-in 5x7 bitmap font.
/// </summary>
/// <remarks>
/// Drawn on the CPU pixels rather than rendered as scene text, so its legibility does not depend
/// on which font shaders the active render pipeline draws in a preview scene.
/// </remarks>
public static class BurnInLabel
{
    private const int GlyphWidth = 5;
    private const int GlyphHeight = 7;
    private const int Margin = 8;

    private static readonly Color32 Ink = new(255, 255, 255, 255);
    private static readonly Color32 Box = new(0, 0, 0, 255);

    // Rows top to bottom; '#' is ink. Only what "frame N" needs.
    private static readonly Dictionary<char, string[]> Glyphs = new()
    {
        ['0'] = new[] { " ### ", "#   #", "#  ##", "# # #", "##  #", "#   #", " ### " },
        ['1'] = new[] { "  #  ", " ##  ", "  #  ", "  #  ", "  #  ", "  #  ", " ### " },
        ['2'] = new[] { " ### ", "#   #", "    #", "   # ", "  #  ", " #   ", "#####" },
        ['3'] = new[] { "#####", "   # ", "  #  ", "   # ", "    #", "#   #", " ### " },
        ['4'] = new[] { "   # ", "  ## ", " # # ", "#  # ", "#####", "   # ", "   # " },
        ['5'] = new[] { "#####", "#    ", "#### ", "    #", "    #", "#   #", " ### " },
        ['6'] = new[] { "  ## ", " #   ", "#    ", "#### ", "#   #", "#   #", " ### " },
        ['7'] = new[] { "#####", "    #", "   # ", "  #  ", " #   ", " #   ", " #   " },
        ['8'] = new[] { " ### ", "#   #", "#   #", " ### ", "#   #", "#   #", " ### " },
        ['9'] = new[] { " ### ", "#   #", "#   #", " ####", "    #", "   # ", " ##  " },
        ['f'] = new[] { "  ## ", " #  #", " #   ", "###  ", " #   ", " #   ", " #   " },
        ['r'] = new[] { "     ", "     ", "# ## ", "##  #", "#    ", "#    ", "#    " },
        ['a'] = new[] { "     ", "     ", " ### ", "    #", " ####", "#   #", " ####" },
        ['m'] = new[] { "     ", "     ", "## # ", "# # #", "# # #", "#   #", "#   #" },
        ['e'] = new[] { "     ", "     ", " ### ", "#   #", "#####", "#    ", " ### " },
        ['-'] = new[] { "     ", "     ", "     ", "#####", "     ", "     ", "     " },
        [' '] = new[] { "     ", "     ", "     ", "     ", "     ", "     ", "     " },
    };

    /// <summary>
    /// Draws <paramref name="text"/> in white on a black box at the image's top-left.
    /// <paramref name="pixels"/> is row-major with row 0 at the bottom, as Unity reads textures back.
    /// Characters with no glyph are drawn as spaces.
    /// </summary>
    public static void Draw(NativeArray<Color32> pixels, int width, int height, string text, int scale)
    {
        if (string.IsNullOrEmpty(text) || scale < 1) return;

        var advance = (GlyphWidth + 1) * scale;
        var boxWidth = text.Length * advance + scale * 3;
        var boxHeight = (GlyphHeight + 4) * scale;

        FillRect(pixels, width, height, Margin, Margin, boxWidth, boxHeight, Box);

        var x = Margin + scale * 2;
        var y = Margin + scale * 2;
        foreach (var character in text)
        {
            if (Glyphs.TryGetValue(character, out var rows)) DrawGlyph(pixels, width, height, rows, x, y, scale);
            x += advance;
        }
    }

    private static void DrawGlyph(NativeArray<Color32> pixels, int width, int height, string[] rows,
        int left, int top, int scale)
    {
        for (var row = 0; row < GlyphHeight; row++)
        {
            for (var column = 0; column < GlyphWidth; column++)
            {
                if (rows[row][column] != '#') continue;

                FillRect(pixels, width, height, left + column * scale, top + row * scale, scale, scale, Ink);
            }
        }
    }

    /// <summary>Fills a rectangle given in top-down image coordinates, clipped to the image.</summary>
    private static void FillRect(NativeArray<Color32> pixels, int width, int height, int left, int top,
        int rectWidth, int rectHeight, Color32 colour)
    {
        var xEnd = Mathf.Min(width, left + rectWidth);
        var yEnd = Mathf.Min(height, top + rectHeight);
        for (var y = Mathf.Max(0, top); y < yEnd; y++)
        {
            var rowStart = (height - 1 - y) * width;
            for (var x = Mathf.Max(0, left); x < xEnd; x++) pixels[rowStart + x] = colour;
        }
    }
}
}
