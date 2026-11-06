"""Record PIL drawing operations for matching-resolution vector PDF export.

Figures are drawn with PIL for the PNG; ``VectorDraw`` records every primitive
as it is drawn so ``save_vector_pdf`` can replay the same scene through the
Matplotlib PDF backend, yielding a PDF with real vector paths and selectable
text instead of an embedded bitmap.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw


class VectorDraw:
    """PIL-compatible drawing proxy that also records vector primitives."""

    def __init__(self, image: Image.Image, scale: float = 1):
        """Wrap ``image``; ``scale`` multiplies all coordinates, widths and fonts."""
        self.image = image
        self.scale = scale
        self._draw = ImageDraw.Draw(image)
        # Keep the original canvas color and the PIL painter's order in the PDF.
        self.operations: list[tuple] = [
            ("rectangle", (0, 0, *image.size), image.getpixel((0, 0)), None, 0)
        ]
        self._fonts = {}

    def __getattr__(self, name):
        # Anything not overridden (textbbox, ...) is delegated to ImageDraw.
        return getattr(self._draw, name)

    def _xy(self, xy):
        """Scale a flat or nested coordinate sequence."""
        return [tuple(v * self.scale for v in p) if isinstance(p, (tuple, list))
                else p * self.scale for p in xy]

    def _width(self, width):
        """Scaled stroke width, at least 1 px."""
        return max(1, round((width or 1) * self.scale))

    def _font(self, font):
        """Scaled copy of ``font`` (cached per path and size)."""
        if self.scale == 1 or font is None:
            return font
        key = (font.path, font.size)
        if key not in self._fonts:
            self._fonts[key] = font.font_variant(size=round(font.size * self.scale))
        return self._fonts[key]

    def line(self, xy, fill=None, width=0, joint=None, **kwargs):
        """Draw and record a polyline."""
        xy, width = self._xy(xy), self._width(width)
        self._draw.line(xy, fill=fill, width=width, joint=joint, **kwargs)
        points = list(xy)
        if points and not isinstance(points[0], (tuple, list)):
            points = list(zip(points[::2], points[1::2]))
        self.operations.append(("line", points, fill, width or 1))

    def ellipse(self, xy, fill=None, outline=None, width=1, **kwargs):
        """Draw and record an ellipse given its bounding box."""
        xy, width = self._xy(xy), self._width(width)
        self._draw.ellipse(xy, fill=fill, outline=outline, width=width, **kwargs)
        self.operations.append(("ellipse", tuple(xy), fill, outline, width))

    def rectangle(self, xy, fill=None, outline=None, width=1, **kwargs):
        """Draw and record an axis-aligned rectangle."""
        xy, width = self._xy(xy), self._width(width)
        self._draw.rectangle(xy, fill=fill, outline=outline, width=width, **kwargs)
        self.operations.append(("rectangle", tuple(xy), fill, outline, width))

    def polygon(self, xy, fill=None, outline=None, width=1, **kwargs):
        """Draw and record a closed polygon."""
        xy, width = self._xy(xy), self._width(width)
        self._draw.polygon(xy, fill=fill, outline=outline, width=width, **kwargs)
        self.operations.append(("polygon", list(xy), fill, outline, width))

    def text(self, xy, text, fill=None, font=None, *args, **kwargs):
        """Draw and record text.

        Extra keywords: ``vector_only`` (record for the PDF but skip the raster,
        used when the raster was pasted separately) and ``rotation`` (degrees).
        """
        vector_only = kwargs.pop("vector_only", False)
        rotation = kwargs.pop("rotation", 0)
        xy, font = self._xy(xy), self._font(font)
        if "spacing" in kwargs:
            kwargs["spacing"] *= self.scale
        if not vector_only:
            self._draw.text(xy, text, fill=fill, font=font, *args, **kwargs)
        self.operations.append(("text", tuple(xy), str(text), fill, font,
                                kwargs.get("spacing", 4 * self.scale), rotation,
                                kwargs.get("anchor", "la")))


def save_vector_pdf(size: tuple[int, int], operations: list[tuple], path: Path,
                    dpi: int = 1000) -> None:
    """Render recorded primitives as PDF paths and text at the requested scale."""
    from matplotlib.backends.backend_pdf import FigureCanvasPdf
    from matplotlib import rc_context
    from matplotlib.figure import Figure
    from matplotlib.font_manager import FontProperties
    from matplotlib.lines import Line2D
    from matplotlib.patches import Ellipse, Polygon, Rectangle

    width, height = size
    # One PIL pixel = one PDF unit at ``dpi``; y axis inverted to match PIL.
    fig = Figure(figsize=(width / dpi, height / dpi), dpi=dpi, facecolor="white")
    FigureCanvasPdf(fig)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_axis_off()

    def color(value):
        """PIL colour (0-255 tuple or name) -> Matplotlib colour."""
        if value is None:
            return "none"
        if isinstance(value, (tuple, list)) and len(value) >= 3:
            channels = value[:3]
            if max(channels) > 1:
                channels = [v / 255 for v in channels]
            return tuple(channels)
        return value

    def add_shape(kind, bounds, face, edge, line_width, zorder):
        """Add an ellipse or rectangle patch from PIL bounding-box coordinates."""
        x0, y0, x1, y1 = map(float, bounds)
        common = {"facecolor": color(face), "edgecolor": color(edge),
                  "linewidth": float(line_width or 1) * 72 / dpi,
                  "clip_on": False, "zorder": zorder}
        if kind == "ellipse":
            artist = Ellipse(((x0 + x1) / 2, (y0 + y1) / 2),
                             abs(x1 - x0), abs(y1 - y0), **common)
        else:
            artist = Rectangle((min(x0, x1), min(y0, y1)), abs(x1 - x0),
                               abs(y1 - y0), **common)
        ax.add_patch(artist)

    # Replay operations in drawing order; zorder preserves PIL's painter order.
    for zorder, operation in enumerate(operations, start=1):
        kind = operation[0]
        if kind == "line":
            _, points, stroke, line_width = operation
            if len(points) < 2:
                continue
            xs, ys = zip(*points)
            ax.add_line(Line2D(xs, ys, color=color(stroke),
                               linewidth=line_width * 72 / dpi,
                               solid_capstyle="round", clip_on=False,
                               zorder=zorder))
        elif kind in ("ellipse", "rectangle"):
            _, bounds, face, edge, line_width = operation
            add_shape(kind, bounds, face, edge, line_width, zorder)
        elif kind == "polygon":
            _, points, face, edge, line_width = operation
            ax.add_patch(Polygon(points, closed=True, facecolor=color(face),
                                 edgecolor=color(edge),
                                 linewidth=line_width * 72 / dpi,
                                 clip_on=False, zorder=zorder))
        elif kind == "text":
            _, (x, y), text, ink, font, spacing, rotation, anchor = operation
            px_size = getattr(font, "size", 12)
            font_path = getattr(font, "path", None)
            prop = (FontProperties(fname=str(font_path), size=px_size * 72 / dpi)
                    if font_path else FontProperties(family="Arial",
                                                     size=px_size * 72 / dpi))
            ascent = font.getmetrics()[0] if hasattr(font, "getmetrics") else px_size
            # PIL's default anchor is the ascender; both renderers share the
            # baseline once this offset is applied. Split multiline text using
            # PIL's own line pitch rather than Matplotlib's different leading.
            if anchor == "la":
                y += ascent
            elif anchor != "ls":
                raise ValueError(f"unsupported vector text anchor: {anchor}")
            pitch = font.getbbox("A")[3] + spacing
            for i, line in enumerate(text.split("\n")):
                ax.text(x, y + i * pitch, line, color=color(ink), fontproperties=prop,
                        ha="left", va="baseline", rotation=rotation,
                        rotation_mode="anchor", clip_on=False, zorder=zorder,
                        parse_math=False)

    path.parent.mkdir(parents=True, exist_ok=True)
    with rc_context({"pdf.fonttype": 42, "pdf.compression": 6}):
        fig.savefig(path, format="pdf", dpi=dpi, facecolor="white",
                    bbox_inches=None, pad_inches=0,
                    metadata={"Creator": "PIL vector export"})
    fig.clear()
