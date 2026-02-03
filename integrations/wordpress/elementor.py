"""Elementor Page Builder - Programmatic design for WordPress.

Creates professional Elementor layouts with sections, columns, and widgets.
Supports modern design patterns: hero sections, feature grids, CTAs, testimonials, etc.

Elementor stores data as JSON in WordPress post meta (`_elementor_data`). The structure is:
- Sections (full-width containers)
  - Columns (divide sections horizontally)
    - Widgets (content: headings, text, images, buttons, etc.)

Each element has:
- `id` - unique 7-char hex identifier
- `elType` - "section", "column", or "widget"
- `widgetType` - for widgets: "heading", "text-editor", "image", "button", etc.
- `settings` - all styling and content
"""

import copy
import json
import random
import string
from typing import Any


def _generate_id() -> str:
    """Generate unique Elementor element ID (7 hex chars)."""
    return ''.join(random.choices(string.hexdigits.lower()[:16], k=7))


def _merge_settings(base: dict, overrides: dict) -> dict:
    """Merge settings dictionaries, handling None values."""
    result = base.copy()
    for key, value in overrides.items():
        if value is not None:
            result[key] = value
    return result


def _create_spacing(
    top: int = None,
    right: int = None,
    bottom: int = None,
    left: int = None,
    unit: str = "px",
    linked: bool = False,
) -> dict:
    """Create Elementor spacing object (padding/margin)."""
    spacing = {
        "unit": unit,
        "isLinked": linked,
    }
    if top is not None:
        spacing["top"] = str(top)
    if right is not None:
        spacing["right"] = str(right)
    if bottom is not None:
        spacing["bottom"] = str(bottom)
    if left is not None:
        spacing["left"] = str(left)
    return spacing


def _create_typography(
    font_family: str = None,
    font_size: dict = None,  # {"size": 16, "unit": "px"}
    font_weight: str = None,  # 100-900, normal, bold
    line_height: dict = None,  # {"size": 1.5, "unit": "em"}
    letter_spacing: dict = None,  # {"size": 1, "unit": "px"}
    text_transform: str = None,  # uppercase, lowercase, capitalize, none
    font_style: str = None,  # normal, italic, oblique
    text_decoration: str = None,  # none, underline, overline, line-through
) -> dict:
    """Create Elementor typography settings."""
    typography = {}
    if font_family:
        typography["typography_font_family"] = font_family
    if font_size:
        typography["typography_font_size"] = font_size
    if font_weight:
        typography["typography_font_weight"] = font_weight
    if line_height:
        typography["typography_line_height"] = line_height
    if letter_spacing:
        typography["typography_letter_spacing"] = letter_spacing
    if text_transform:
        typography["typography_text_transform"] = text_transform
    if font_style:
        typography["typography_font_style"] = font_style
    if text_decoration:
        typography["typography_text_decoration"] = text_decoration
    return typography


# ============ Core Element Builders ============

def create_section(
    columns: list[dict],
    layout: str = "boxed",  # boxed, full_width
    stretch: bool = False,
    gap: str = "default",  # no, narrow, default, wide, wider, extended
    content_width: str = "boxed",  # boxed, full
    min_height: dict = None,  # {"size": 100, "unit": "vh"}
    background: dict = None,  # See background helpers
    padding: dict = None,
    margin: dict = None,
    css_classes: str = "",
    z_index: int = None,
    overflow: str = None,  # hidden, visible
    html_tag: str = "section",  # section, header, footer, main, article, aside, nav
) -> dict:
    """Create an Elementor section with columns.

    Args:
        columns: List of column elements created with create_column()
        layout: Section layout - "boxed" or "full_width"
        stretch: Whether to stretch section to full width
        gap: Gap between columns
        content_width: Content area width
        min_height: Minimum height with size and unit
        background: Background settings from background helpers
        padding: Padding settings
        margin: Margin settings
        css_classes: Additional CSS classes
        z_index: Z-index for layering
        overflow: Overflow behavior
        html_tag: HTML tag to use for section

    Returns:
        Elementor section element dict
    """
    settings = {
        "layout": layout,
        "gap": gap,
        "content_width": content_width,
        "structure": _get_column_structure(len(columns)),
    }

    if stretch:
        settings["stretch_section"] = "section-stretched"

    if min_height:
        settings["custom_height"] = "min-height"
        settings["custom_height_inner"] = min_height

    if background:
        settings.update(background)

    if padding:
        settings["padding"] = padding

    if margin:
        settings["margin"] = margin

    if css_classes:
        settings["css_classes"] = css_classes

    if z_index is not None:
        settings["z_index"] = z_index

    if overflow:
        settings["overflow"] = overflow

    if html_tag != "section":
        settings["html_tag"] = html_tag

    return {
        "id": _generate_id(),
        "elType": "section",
        "settings": settings,
        "elements": columns,
        "isInner": False,
    }


def create_inner_section(
    columns: list[dict],
    gap: str = "default",
    background: dict = None,
    padding: dict = None,
    margin: dict = None,
    css_classes: str = "",
) -> dict:
    """Create an inner/nested section (for complex layouts).

    Args:
        columns: List of column elements
        gap: Gap between columns
        background: Background settings
        padding: Padding settings
        margin: Margin settings
        css_classes: Additional CSS classes

    Returns:
        Elementor inner section element dict
    """
    settings = {
        "gap": gap,
        "structure": _get_column_structure(len(columns)),
    }

    if background:
        settings.update(background)

    if padding:
        settings["padding"] = padding

    if margin:
        settings["margin"] = margin

    if css_classes:
        settings["css_classes"] = css_classes

    return {
        "id": _generate_id(),
        "elType": "section",
        "settings": settings,
        "elements": columns,
        "isInner": True,
    }


def _get_column_structure(num_columns: int) -> str:
    """Get Elementor structure code for number of columns."""
    structures = {
        1: "10",
        2: "20",
        3: "30",
        4: "40",
        5: "50",
        6: "60",
    }
    return structures.get(num_columns, "10")


def create_column(
    widgets: list[dict],
    width: int = 100,  # Percentage width
    vertical_align: str = "top",  # top, middle, bottom, space-between, space-around, space-evenly
    horizontal_align: str = "default",  # default, start, center, end, space-between, space-around, space-evenly
    background: dict = None,
    padding: dict = None,
    margin: dict = None,
    css_classes: str = "",
    z_index: int = None,
    html_tag: str = "div",
) -> dict:
    """Create an Elementor column with widgets.

    Args:
        widgets: List of widget elements
        width: Column width as percentage (e.g., 50 for 50%)
        vertical_align: Vertical alignment of content
        horizontal_align: Horizontal alignment of content
        background: Background settings
        padding: Padding settings
        margin: Margin settings
        css_classes: Additional CSS classes
        z_index: Z-index for layering
        html_tag: HTML tag to use

    Returns:
        Elementor column element dict
    """
    # Elementor stores column size as percentage * 10 (e.g., 50% = 500)
    # Actually it's just the percentage (50 for 50%)
    settings = {
        "_column_size": width,
    }

    if vertical_align != "top":
        settings["_inline_size"] = None
        settings["content_position"] = vertical_align

    if horizontal_align != "default":
        settings["_inline_size"] = None
        settings["align"] = horizontal_align

    if background:
        settings.update(background)

    if padding:
        settings["padding"] = padding

    if margin:
        settings["margin"] = margin

    if css_classes:
        settings["css_classes"] = css_classes

    if z_index is not None:
        settings["z_index"] = z_index

    if html_tag != "div":
        settings["html_tag"] = html_tag

    return {
        "id": _generate_id(),
        "elType": "column",
        "settings": settings,
        "elements": widgets,
        "isInner": False,
    }


# ============ Widget Builders ============

def widget_heading(
    text: str,
    tag: str = "h2",  # h1-h6, div, span, p
    align: str = "left",  # left, center, right, justify
    size: str = "default",  # small, medium, large, xl, xxl
    color: str = None,  # hex color
    typography: dict = None,  # font_family, font_size, font_weight, etc.
    link: str = None,
    css_classes: str = "",
) -> dict:
    """Create heading widget.

    Args:
        text: Heading text content
        tag: HTML tag (h1-h6, div, span, p)
        align: Text alignment
        size: Predefined size class
        color: Text color (hex)
        typography: Typography settings from _create_typography()
        link: Optional link URL
        css_classes: Additional CSS classes

    Returns:
        Elementor heading widget dict
    """
    settings = {
        "title": text,
        "header_size": tag,
        "align": align,
        "size": size,
    }

    if color:
        settings["title_color"] = color

    if typography:
        settings["typography_typography"] = "custom"
        settings.update(typography)

    if link:
        settings["link"] = {
            "url": link,
            "is_external": link.startswith("http") and "://" in link,
            "nofollow": False,
        }

    if css_classes:
        settings["css_classes"] = css_classes

    return {
        "id": _generate_id(),
        "elType": "widget",
        "widgetType": "heading",
        "settings": settings,
        "elements": [],
    }


def widget_text(
    content: str,  # HTML content
    align: str = "left",
    color: str = None,
    typography: dict = None,
    css_classes: str = "",
) -> dict:
    """Create text editor widget.

    Args:
        content: HTML content for the text
        align: Text alignment
        color: Text color (hex)
        typography: Typography settings
        css_classes: Additional CSS classes

    Returns:
        Elementor text-editor widget dict
    """
    settings = {
        "editor": content,
        "align": align,
    }

    if color:
        settings["text_color"] = color

    if typography:
        settings["typography_typography"] = "custom"
        settings.update(typography)

    if css_classes:
        settings["css_classes"] = css_classes

    return {
        "id": _generate_id(),
        "elType": "widget",
        "widgetType": "text-editor",
        "settings": settings,
        "elements": [],
    }


def widget_image(
    url: str,
    alt: str = "",
    size: str = "full",  # thumbnail, medium, large, full, custom
    align: str = "center",  # left, center, right
    link: str = None,
    link_target: str = "_self",  # _self, _blank
    caption: str = "",
    width: dict = None,  # {"size": 100, "unit": "%"}
    max_width: dict = None,
    border_radius: dict = None,
    css_classes: str = "",
    hover_animation: str = None,  # grow, shrink, pulse, push, etc.
) -> dict:
    """Create image widget.

    Args:
        url: Image URL
        alt: Alt text for accessibility
        size: Image size preset
        align: Image alignment
        link: Optional link URL
        link_target: Link target (_self or _blank)
        caption: Image caption
        width: Custom width
        max_width: Maximum width
        border_radius: Border radius settings
        css_classes: Additional CSS classes
        hover_animation: Hover animation effect

    Returns:
        Elementor image widget dict
    """
    settings = {
        "image": {
            "url": url,
            "id": "",
            "alt": alt,
            "source": "library",
        },
        "image_size": size,
        "align": align,
    }

    if link:
        settings["link_to"] = "custom"
        settings["link"] = {
            "url": link,
            "is_external": link_target == "_blank",
            "nofollow": False,
        }
    else:
        settings["link_to"] = "none"

    if caption:
        settings["caption_source"] = "custom"
        settings["caption"] = caption

    if width:
        settings["width"] = width

    if max_width:
        settings["width_px"] = max_width

    if border_radius:
        settings["image_border_radius"] = border_radius

    if css_classes:
        settings["css_classes"] = css_classes

    if hover_animation:
        settings["hover_animation"] = hover_animation

    return {
        "id": _generate_id(),
        "elType": "widget",
        "widgetType": "image",
        "settings": settings,
        "elements": [],
    }


def widget_button(
    text: str,
    link: str = "#",
    style: str = "default",  # default, info, success, warning, danger
    size: str = "md",  # xs, sm, md, lg, xl
    align: str = "center",  # left, center, right, justify
    icon: str = None,  # FontAwesome class (e.g., "fa fa-arrow-right")
    icon_position: str = "left",  # left, right
    full_width: bool = False,
    link_target: str = "_self",
    css_classes: str = "",
    button_css_id: str = "",
    hover_animation: str = None,
    background_color: str = None,
    text_color: str = None,
    border_radius: dict = None,
) -> dict:
    """Create button widget.

    Args:
        text: Button text
        link: Button link URL
        style: Button style preset
        size: Button size
        align: Button alignment
        icon: FontAwesome icon class
        icon_position: Icon position (left or right)
        full_width: Whether button spans full width
        link_target: Link target
        css_classes: Additional CSS classes
        button_css_id: CSS ID for the button
        hover_animation: Hover animation effect
        background_color: Custom background color
        text_color: Custom text color
        border_radius: Custom border radius

    Returns:
        Elementor button widget dict
    """
    settings = {
        "text": text,
        "link": {
            "url": link,
            "is_external": link_target == "_blank",
            "nofollow": False,
        },
        "button_type": style,
        "size": size,
        "align": align,
    }

    if icon:
        settings["selected_icon"] = {
            "value": icon,
            "library": "fa-solid" if icon.startswith("fas ") else "fa-regular" if icon.startswith("far ") else "solid",
        }
        settings["icon_align"] = icon_position

    if full_width:
        settings["button_width"] = "100"

    if css_classes:
        settings["css_classes"] = css_classes

    if button_css_id:
        settings["button_css_id"] = button_css_id

    if hover_animation:
        settings["hover_animation"] = hover_animation

    if background_color:
        settings["background_color"] = background_color

    if text_color:
        settings["button_text_color"] = text_color

    if border_radius:
        settings["border_radius"] = border_radius

    return {
        "id": _generate_id(),
        "elType": "widget",
        "widgetType": "button",
        "settings": settings,
        "elements": [],
    }


def widget_icon_list(
    items: list[dict],  # [{"text": "Item 1", "icon": "fa fa-check", "link": ""}]
    icon_color: str = None,
    text_color: str = None,
    space_between: int = None,
    icon_size: dict = None,  # {"size": 14, "unit": "px"}
    text_indent: dict = None,  # {"size": 10, "unit": "px"}
    divider: bool = False,
    divider_style: str = "solid",
    divider_color: str = None,
    css_classes: str = "",
) -> dict:
    """Create icon list widget.

    Args:
        items: List of items with text, icon, and optional link
        icon_color: Color for all icons
        text_color: Color for all text
        space_between: Space between items in pixels
        icon_size: Icon size settings
        text_indent: Text indent from icon
        divider: Whether to show dividers between items
        divider_style: Divider line style
        divider_color: Divider color
        css_classes: Additional CSS classes

    Returns:
        Elementor icon-list widget dict
    """
    icon_list = []
    for item in items:
        list_item = {
            "text": item.get("text", ""),
            "selected_icon": {
                "value": item.get("icon", "fas fa-check"),
                "library": "fa-solid",
            },
        }
        if item.get("link"):
            list_item["link"] = {
                "url": item["link"],
                "is_external": False,
                "nofollow": False,
            }
        icon_list.append(list_item)

    settings = {
        "icon_list": icon_list,
    }

    if icon_color:
        settings["icon_color"] = icon_color

    if text_color:
        settings["text_color"] = text_color

    if space_between is not None:
        settings["space_between"] = {"size": space_between, "unit": "px"}

    if icon_size:
        settings["icon_size"] = icon_size

    if text_indent:
        settings["text_indent"] = text_indent

    if divider:
        settings["divider"] = "yes"
        settings["divider_style"] = divider_style
        if divider_color:
            settings["divider_color"] = divider_color

    if css_classes:
        settings["css_classes"] = css_classes

    return {
        "id": _generate_id(),
        "elType": "widget",
        "widgetType": "icon-list",
        "settings": settings,
        "elements": [],
    }


def widget_spacer(
    height: int = 50,
    unit: str = "px",
    responsive: dict = None,  # {"tablet": 30, "mobile": 20}
) -> dict:
    """Create spacer widget.

    Args:
        height: Spacer height
        unit: Height unit (px, em, vh)
        responsive: Responsive height overrides for tablet/mobile

    Returns:
        Elementor spacer widget dict
    """
    settings = {
        "space": {
            "size": height,
            "unit": unit,
        },
    }

    if responsive:
        if "tablet" in responsive:
            settings["space_tablet"] = {"size": responsive["tablet"], "unit": unit}
        if "mobile" in responsive:
            settings["space_mobile"] = {"size": responsive["mobile"], "unit": unit}

    return {
        "id": _generate_id(),
        "elType": "widget",
        "widgetType": "spacer",
        "settings": settings,
        "elements": [],
    }


def widget_divider(
    style: str = "solid",  # solid, double, dotted, dashed
    weight: int = 1,
    color: str = None,
    width: int = 100,  # percentage
    align: str = "center",  # left, center, right
    gap: dict = None,  # {"size": 15, "unit": "px"} - space above/below
    element: str = None,  # none, text, icon
    element_text: str = None,
    element_icon: str = None,
    css_classes: str = "",
) -> dict:
    """Create divider widget.

    Args:
        style: Line style
        weight: Line thickness in pixels
        color: Line color (hex)
        width: Width as percentage
        align: Alignment
        gap: Space above and below
        element: Optional element in divider (text or icon)
        element_text: Text if element="text"
        element_icon: Icon class if element="icon"
        css_classes: Additional CSS classes

    Returns:
        Elementor divider widget dict
    """
    settings = {
        "style": style,
        "weight": {"size": weight, "unit": "px"},
        "width": {"size": width, "unit": "%"},
        "align": align,
    }

    if color:
        settings["color"] = color

    if gap:
        settings["gap"] = gap

    if element and element != "none":
        settings["look"] = element
        if element == "text" and element_text:
            settings["text"] = element_text
        elif element == "icon" and element_icon:
            settings["selected_icon"] = {
                "value": element_icon,
                "library": "fa-solid",
            }

    if css_classes:
        settings["css_classes"] = css_classes

    return {
        "id": _generate_id(),
        "elType": "widget",
        "widgetType": "divider",
        "settings": settings,
        "elements": [],
    }


def widget_video(
    url: str,  # YouTube, Vimeo, or self-hosted URL
    video_type: str = "youtube",  # youtube, vimeo, hosted, dailymotion
    autoplay: bool = False,
    mute: bool = False,
    loop: bool = False,
    controls: bool = True,
    start_time: int = None,
    end_time: int = None,
    poster_url: str = None,  # Custom thumbnail
    lazy_load: bool = True,
    aspect_ratio: str = "169",  # 169, 219, 43, 32, 11, 916
    css_classes: str = "",
) -> dict:
    """Create video widget.

    Args:
        url: Video URL (YouTube, Vimeo, or direct video file)
        video_type: Type of video source
        autoplay: Auto-play on load
        mute: Start muted
        loop: Loop video
        controls: Show video controls
        start_time: Start time in seconds
        end_time: End time in seconds
        poster_url: Custom thumbnail/poster image
        lazy_load: Enable lazy loading
        aspect_ratio: Video aspect ratio
        css_classes: Additional CSS classes

    Returns:
        Elementor video widget dict
    """
    settings = {
        "video_type": video_type,
        "youtube_url": url if video_type == "youtube" else "",
        "vimeo_url": url if video_type == "vimeo" else "",
        "dailymotion_url": url if video_type == "dailymotion" else "",
        "aspect_ratio": aspect_ratio,
    }

    if video_type == "hosted":
        settings["hosted_url"] = {"url": url, "id": ""}

    if autoplay:
        settings["autoplay"] = "yes"

    if mute:
        settings["mute"] = "yes"

    if loop:
        settings["loop"] = "yes"

    if not controls:
        settings["controls"] = ""
    else:
        settings["controls"] = "yes"

    if start_time:
        settings["start"] = start_time

    if end_time:
        settings["end"] = end_time

    if poster_url:
        settings["image_overlay"] = {"url": poster_url, "id": ""}
        settings["show_image_overlay"] = "yes"

    if lazy_load:
        settings["lazy_load"] = "yes"

    if css_classes:
        settings["css_classes"] = css_classes

    return {
        "id": _generate_id(),
        "elType": "widget",
        "widgetType": "video",
        "settings": settings,
        "elements": [],
    }


def widget_icon_box(
    title: str,
    description: str,
    icon: str = "fas fa-star",
    icon_position: str = "top",  # top, left, right
    icon_color: str = None,
    icon_size: dict = None,  # {"size": 50, "unit": "px"}
    title_color: str = None,
    description_color: str = None,
    link: str = None,
    link_target: str = "_self",
    title_tag: str = "h3",
    title_spacing: dict = None,
    css_classes: str = "",
) -> dict:
    """Create icon box widget.

    Args:
        title: Box title
        description: Box description text
        icon: FontAwesome icon class
        icon_position: Icon position relative to content
        icon_color: Icon color (hex)
        icon_size: Icon size settings
        title_color: Title color (hex)
        description_color: Description color (hex)
        link: Optional link URL
        link_target: Link target
        title_tag: HTML tag for title
        title_spacing: Space below title
        css_classes: Additional CSS classes

    Returns:
        Elementor icon-box widget dict
    """
    settings = {
        "selected_icon": {
            "value": icon,
            "library": "fa-solid",
        },
        "title_text": title,
        "description_text": description,
        "position": icon_position,
        "title_size": title_tag,
    }

    if icon_color:
        settings["primary_color"] = icon_color

    if icon_size:
        settings["icon_size"] = icon_size

    if title_color:
        settings["title_color"] = title_color

    if description_color:
        settings["description_color"] = description_color

    if link:
        settings["link"] = {
            "url": link,
            "is_external": link_target == "_blank",
            "nofollow": False,
        }

    if title_spacing:
        settings["title_bottom_space"] = title_spacing

    if css_classes:
        settings["css_classes"] = css_classes

    return {
        "id": _generate_id(),
        "elType": "widget",
        "widgetType": "icon-box",
        "settings": settings,
        "elements": [],
    }


def widget_testimonial(
    content: str,
    name: str,
    title: str = "",
    image_url: str = None,
    align: str = "center",  # left, center, right
    image_position: str = "aside",  # aside, top
    image_size: dict = None,  # {"size": 60, "unit": "px"}
    content_color: str = None,
    name_color: str = None,
    title_color: str = None,
    css_classes: str = "",
) -> dict:
    """Create testimonial widget.

    Args:
        content: Testimonial text/quote
        name: Person's name
        title: Person's title/role
        image_url: Profile image URL
        align: Content alignment
        image_position: Image position
        image_size: Image size settings
        content_color: Quote text color
        name_color: Name color
        title_color: Title color
        css_classes: Additional CSS classes

    Returns:
        Elementor testimonial widget dict
    """
    settings = {
        "testimonial_content": content,
        "testimonial_name": name,
        "testimonial_job": title,
        "testimonial_alignment": align,
        "testimonial_image_position": image_position,
    }

    if image_url:
        settings["testimonial_image"] = {
            "url": image_url,
            "id": "",
        }

    if image_size:
        settings["testimonial_image_size"] = image_size

    if content_color:
        settings["content_text_color"] = content_color

    if name_color:
        settings["name_text_color"] = name_color

    if title_color:
        settings["job_text_color"] = title_color

    if css_classes:
        settings["css_classes"] = css_classes

    return {
        "id": _generate_id(),
        "elType": "widget",
        "widgetType": "testimonial",
        "settings": settings,
        "elements": [],
    }


def widget_counter(
    number: int,
    title: str = "",
    prefix: str = "",
    suffix: str = "",
    duration: int = 2000,  # Animation duration in ms
    thousand_separator: str = ",",
    align: str = "center",
    number_color: str = None,
    title_color: str = None,
    number_size: dict = None,
    css_classes: str = "",
) -> dict:
    """Create counter/number widget.

    Args:
        number: The number to count to
        title: Label below the number
        prefix: Text before number
        suffix: Text after number
        duration: Animation duration in milliseconds
        thousand_separator: Separator for thousands
        align: Alignment
        number_color: Number color (hex)
        title_color: Title color (hex)
        number_size: Number font size
        css_classes: Additional CSS classes

    Returns:
        Elementor counter widget dict
    """
    settings = {
        "ending_number": number,
        "title": title,
        "prefix": prefix,
        "suffix": suffix,
        "duration": duration,
        "thousand_separator": "yes" if thousand_separator else "",
        "thousand_separator_char": thousand_separator,
        "counter_alignment": align,
    }

    if number_color:
        settings["number_color"] = number_color

    if title_color:
        settings["title_color"] = title_color

    if number_size:
        settings["number_size"] = number_size

    if css_classes:
        settings["css_classes"] = css_classes

    return {
        "id": _generate_id(),
        "elType": "widget",
        "widgetType": "counter",
        "settings": settings,
        "elements": [],
    }


def widget_progress_bar(
    title: str,
    percentage: int,
    color: str = None,
    background_color: str = None,
    show_percentage: bool = True,
    bar_height: dict = None,  # {"size": 30, "unit": "px"}
    title_color: str = None,
    inner_text: str = None,
    css_classes: str = "",
) -> dict:
    """Create progress bar widget.

    Args:
        title: Bar label/title
        percentage: Progress percentage (0-100)
        color: Bar fill color
        background_color: Bar background color
        show_percentage: Whether to show percentage text
        bar_height: Bar height settings
        title_color: Title color
        inner_text: Text inside the bar
        css_classes: Additional CSS classes

    Returns:
        Elementor progress widget dict
    """
    settings = {
        "title": title,
        "percent": {"size": percentage, "unit": "%"},
        "display_percentage": "show" if show_percentage else "hide",
    }

    if color:
        settings["bar_color"] = color

    if background_color:
        settings["bar_bg_color"] = background_color

    if bar_height:
        settings["bar_height"] = bar_height

    if title_color:
        settings["title_color"] = title_color

    if inner_text:
        settings["inner_text"] = inner_text

    if css_classes:
        settings["css_classes"] = css_classes

    return {
        "id": _generate_id(),
        "elType": "widget",
        "widgetType": "progress",
        "settings": settings,
        "elements": [],
    }


def widget_google_maps(
    address: str,
    zoom: int = 14,
    height: int = 300,
    marker_title: str = "",
    marker_description: str = "",
    prevent_scroll: bool = True,
    css_classes: str = "",
) -> dict:
    """Create Google Maps widget.

    Args:
        address: Location address to display
        zoom: Map zoom level (1-20)
        height: Map height in pixels
        marker_title: Marker popup title
        marker_description: Marker popup description
        prevent_scroll: Prevent scroll zoom
        css_classes: Additional CSS classes

    Returns:
        Elementor google_maps widget dict
    """
    settings = {
        "address": address,
        "zoom": {"size": zoom, "unit": "px"},
        "height": {"size": height, "unit": "px"},
    }

    if marker_title:
        settings["marker_title"] = marker_title

    if marker_description:
        settings["marker_description"] = marker_description

    if prevent_scroll:
        settings["prevent_scroll"] = "yes"

    if css_classes:
        settings["css_classes"] = css_classes

    return {
        "id": _generate_id(),
        "elType": "widget",
        "widgetType": "google_maps",
        "settings": settings,
        "elements": [],
    }


def widget_form(
    fields: list[dict],  # [{"type": "text", "label": "Name", "required": True, "placeholder": ""}]
    submit_text: str = "Submit",
    email_to: str = None,
    form_name: str = "Contact Form",
    button_size: str = "md",
    button_width: str = "",  # "", "100" for full width
    button_align: str = "stretch",
    success_message: str = "Your message was sent successfully.",
    error_message: str = "An error occurred. Please try again.",
    required_mark: bool = True,
    css_classes: str = "",
) -> dict:
    """Create form widget (requires Elementor Pro).

    Args:
        fields: List of form fields with type, label, required, placeholder
        submit_text: Submit button text
        email_to: Email address for form submissions
        form_name: Form name for admin reference
        button_size: Submit button size
        button_width: Submit button width
        button_align: Submit button alignment
        success_message: Success message after submission
        error_message: Error message on failure
        required_mark: Show asterisk for required fields
        css_classes: Additional CSS classes

    Returns:
        Elementor form widget dict (Pro feature)
    """
    form_fields = []
    for idx, field in enumerate(fields):
        field_id = f"field_{_generate_id()[:5]}"
        form_field = {
            "_id": field_id,
            "field_type": field.get("type", "text"),
            "field_label": field.get("label", ""),
            "required": "yes" if field.get("required", False) else "",
            "placeholder": field.get("placeholder", ""),
            "field_html": field.get("html", ""),
            "width": field.get("width", "100"),  # Percentage width
        }

        # Handle specific field types
        if field.get("type") == "select" and field.get("options"):
            form_field["field_options"] = "\n".join(field["options"])
        elif field.get("type") == "textarea":
            form_field["rows"] = field.get("rows", 5)
        elif field.get("type") == "acceptance":
            form_field["acceptance_text"] = field.get("acceptance_text", "I agree to the terms.")

        form_fields.append(form_field)

    settings = {
        "form_name": form_name,
        "form_fields": form_fields,
        "button_text": submit_text,
        "button_size": button_size,
        "button_align": button_align,
        "success_message": success_message,
        "error_message": error_message,
        "show_required_mark": "yes" if required_mark else "",
    }

    if button_width:
        settings["button_width"] = button_width

    if email_to:
        settings["email_to"] = email_to
        settings["submit_actions"] = ["email"]

    if css_classes:
        settings["css_classes"] = css_classes

    return {
        "id": _generate_id(),
        "elType": "widget",
        "widgetType": "form",
        "settings": settings,
        "elements": [],
    }


def widget_html(
    code: str,
    css_classes: str = "",
) -> dict:
    """Create custom HTML widget.

    Args:
        code: Raw HTML code
        css_classes: Additional CSS classes

    Returns:
        Elementor html widget dict
    """
    settings = {
        "html": code,
    }

    if css_classes:
        settings["css_classes"] = css_classes

    return {
        "id": _generate_id(),
        "elType": "widget",
        "widgetType": "html",
        "settings": settings,
        "elements": [],
    }


def widget_accordion(
    items: list[dict],  # [{"title": "Question", "content": "Answer"}]
    icon: str = "fas fa-plus",
    active_icon: str = "fas fa-minus",
    icon_align: str = "right",  # left, right
    default_open: int = 0,  # Index of default open item, -1 for all closed
    title_color: str = None,
    content_color: str = None,
    css_classes: str = "",
) -> dict:
    """Create accordion widget.

    Args:
        items: List of accordion items with title and content
        icon: Closed state icon
        active_icon: Open state icon
        icon_align: Icon alignment
        default_open: Index of item to open by default (-1 for none)
        title_color: Title text color
        content_color: Content text color
        css_classes: Additional CSS classes

    Returns:
        Elementor accordion widget dict
    """
    tabs = []
    for item in items:
        tabs.append({
            "_id": _generate_id()[:8],
            "tab_title": item.get("title", ""),
            "tab_content": item.get("content", ""),
        })

    settings = {
        "tabs": tabs,
        "selected_icon": {
            "value": icon,
            "library": "fa-solid",
        },
        "selected_active_icon": {
            "value": active_icon,
            "library": "fa-solid",
        },
        "icon_align": icon_align,
    }

    if title_color:
        settings["title_color"] = title_color

    if content_color:
        settings["content_color"] = content_color

    if css_classes:
        settings["css_classes"] = css_classes

    return {
        "id": _generate_id(),
        "elType": "widget",
        "widgetType": "accordion",
        "settings": settings,
        "elements": [],
    }


def widget_tabs(
    items: list[dict],  # [{"title": "Tab 1", "content": "Content"}]
    tab_type: str = "horizontal",  # horizontal, vertical
    default_active: int = 0,
    title_color: str = None,
    active_title_color: str = None,
    content_color: str = None,
    css_classes: str = "",
) -> dict:
    """Create tabs widget.

    Args:
        items: List of tab items with title and content
        tab_type: Tab layout type
        default_active: Index of default active tab
        title_color: Inactive tab title color
        active_title_color: Active tab title color
        content_color: Content text color
        css_classes: Additional CSS classes

    Returns:
        Elementor tabs widget dict
    """
    tabs = []
    for item in items:
        tabs.append({
            "_id": _generate_id()[:8],
            "tab_title": item.get("title", ""),
            "tab_content": item.get("content", ""),
        })

    settings = {
        "tabs": tabs,
        "type": tab_type,
    }

    if title_color:
        settings["tab_color"] = title_color

    if active_title_color:
        settings["tab_active_color"] = active_title_color

    if content_color:
        settings["content_color"] = content_color

    if css_classes:
        settings["css_classes"] = css_classes

    return {
        "id": _generate_id(),
        "elType": "widget",
        "widgetType": "tabs",
        "settings": settings,
        "elements": [],
    }


def widget_alert(
    title: str,
    description: str = "",
    alert_type: str = "info",  # info, success, warning, danger
    show_dismiss: bool = True,
    css_classes: str = "",
) -> dict:
    """Create alert widget.

    Args:
        title: Alert title
        description: Alert description
        alert_type: Alert type/style
        show_dismiss: Show dismiss button
        css_classes: Additional CSS classes

    Returns:
        Elementor alert widget dict
    """
    settings = {
        "alert_title": title,
        "alert_description": description,
        "alert_type": alert_type,
        "show_dismiss": "show" if show_dismiss else "hide",
    }

    if css_classes:
        settings["css_classes"] = css_classes

    return {
        "id": _generate_id(),
        "elType": "widget",
        "widgetType": "alert",
        "settings": settings,
        "elements": [],
    }


def widget_social_icons(
    icons: list[dict],  # [{"platform": "facebook", "link": "https://..."}]
    shape: str = "rounded",  # rounded, square, circle
    color: str = "official",  # official, custom
    custom_color: str = None,
    size: dict = None,
    spacing: dict = None,
    align: str = "center",
    css_classes: str = "",
) -> dict:
    """Create social icons widget.

    Args:
        icons: List of social icons with platform and link
        shape: Icon shape style
        color: Color mode (official or custom)
        custom_color: Custom color if color="custom"
        size: Icon size settings
        spacing: Space between icons
        align: Icons alignment
        css_classes: Additional CSS classes

    Returns:
        Elementor social-icons widget dict
    """
    social_icon_list = []
    for icon in icons:
        platform = icon.get("platform", "facebook")
        social_icon_list.append({
            "_id": _generate_id()[:8],
            "social_icon": {
                "value": f"fab fa-{platform}",
                "library": "fa-brands",
            },
            "link": {
                "url": icon.get("link", "#"),
                "is_external": True,
                "nofollow": False,
            },
        })

    settings = {
        "social_icon_list": social_icon_list,
        "shape": shape,
        "icon_color": color,
        "align": align,
    }

    if color == "custom" and custom_color:
        settings["icon_primary_color"] = custom_color

    if size:
        settings["icon_size"] = size

    if spacing:
        settings["icon_spacing"] = spacing

    if css_classes:
        settings["css_classes"] = css_classes

    return {
        "id": _generate_id(),
        "elType": "widget",
        "widgetType": "social-icons",
        "settings": settings,
        "elements": [],
    }


def widget_star_rating(
    rating: float = 5,
    scale: int = 5,
    title: str = "",
    align: str = "center",
    star_color: str = None,
    unmarked_star_color: str = None,
    star_size: dict = None,
    css_classes: str = "",
) -> dict:
    """Create star rating widget.

    Args:
        rating: Rating value
        scale: Rating scale (max stars)
        title: Optional title before stars
        align: Alignment
        star_color: Filled star color
        unmarked_star_color: Empty star color
        star_size: Star size settings
        css_classes: Additional CSS classes

    Returns:
        Elementor star-rating widget dict
    """
    settings = {
        "rating_scale": scale,
        "rating": rating,
        "star_title": title,
        "star_align": align,
    }

    if star_color:
        settings["star_color"] = star_color

    if unmarked_star_color:
        settings["unmarked_star_color"] = unmarked_star_color

    if star_size:
        settings["star_size"] = star_size

    if css_classes:
        settings["css_classes"] = css_classes

    return {
        "id": _generate_id(),
        "elType": "widget",
        "widgetType": "star-rating",
        "settings": settings,
        "elements": [],
    }


def widget_image_carousel(
    images: list[dict],  # [{"url": "...", "alt": "...", "link": "..."}]
    slides_to_show: int = 3,
    slides_to_scroll: int = 1,
    navigation: str = "both",  # arrows, dots, both, none
    autoplay: bool = True,
    autoplay_speed: int = 5000,
    infinite_loop: bool = True,
    pause_on_hover: bool = True,
    image_stretch: str = "no",  # yes, no
    css_classes: str = "",
) -> dict:
    """Create image carousel widget.

    Args:
        images: List of images with url, alt, and optional link
        slides_to_show: Number of slides visible at once
        slides_to_scroll: Number of slides to scroll
        navigation: Navigation style
        autoplay: Enable autoplay
        autoplay_speed: Autoplay interval in ms
        infinite_loop: Loop infinitely
        pause_on_hover: Pause autoplay on hover
        image_stretch: Stretch images to fit
        css_classes: Additional CSS classes

    Returns:
        Elementor image-carousel widget dict
    """
    carousel_slides = []
    for img in images:
        slide = {
            "_id": _generate_id()[:8],
            "image": {
                "url": img.get("url", ""),
                "id": "",
            },
        }
        if img.get("link"):
            slide["link_to"] = "custom"
            slide["link"] = {"url": img["link"], "is_external": False, "nofollow": False}
        carousel_slides.append(slide)

    settings = {
        "carousel": carousel_slides,
        "slides_to_show": str(slides_to_show),
        "slides_to_scroll": str(slides_to_scroll),
        "image_stretch": image_stretch,
        "navigation": navigation,
        "autoplay": "yes" if autoplay else "",
        "autoplay_speed": autoplay_speed,
        "infinite": "yes" if infinite_loop else "",
        "pause_on_hover": "yes" if pause_on_hover else "",
    }

    if css_classes:
        settings["css_classes"] = css_classes

    return {
        "id": _generate_id(),
        "elType": "widget",
        "widgetType": "image-carousel",
        "settings": settings,
        "elements": [],
    }


# ============ Background Helpers ============

def background_color(color: str) -> dict:
    """Solid color background.

    Args:
        color: Hex color code

    Returns:
        Background settings dict
    """
    return {
        "background_background": "classic",
        "background_color": color,
    }


def background_gradient(
    color1: str,
    color2: str,
    angle: int = 180,
    gradient_type: str = "linear",  # linear, radial
    position: str = "center center",
) -> dict:
    """Gradient background.

    Args:
        color1: First gradient color (hex)
        color2: Second gradient color (hex)
        angle: Gradient angle in degrees (for linear)
        gradient_type: Gradient type
        position: Gradient position (for radial)

    Returns:
        Background settings dict
    """
    settings = {
        "background_background": "gradient",
        "background_color": color1,
        "background_color_b": color2,
        "background_gradient_type": gradient_type,
    }

    if gradient_type == "linear":
        settings["background_gradient_angle"] = {"size": angle, "unit": "deg"}
    else:
        settings["background_gradient_position"] = position

    return settings


def background_image(
    url: str,
    position: str = "center center",
    size: str = "cover",  # cover, contain, auto
    repeat: str = "no-repeat",  # no-repeat, repeat, repeat-x, repeat-y
    attachment: str = "scroll",  # scroll, fixed
    overlay_color: str = None,
    overlay_opacity: float = 0.5,
) -> dict:
    """Image background with optional overlay.

    Args:
        url: Image URL
        position: Background position
        size: Background size
        repeat: Background repeat
        attachment: Background attachment
        overlay_color: Overlay color (hex)
        overlay_opacity: Overlay opacity (0-1)

    Returns:
        Background settings dict
    """
    settings = {
        "background_background": "classic",
        "background_image": {
            "url": url,
            "id": "",
        },
        "background_position": position,
        "background_size": size,
        "background_repeat": repeat,
        "background_attachment": attachment,
    }

    if overlay_color:
        settings["background_overlay_background"] = "classic"
        settings["background_overlay_color"] = overlay_color
        settings["background_overlay_opacity"] = {"size": overlay_opacity, "unit": ""}

    return settings


def background_video(
    url: str,
    fallback_image: str = None,
    play_once: bool = False,
    play_on_mobile: bool = False,
) -> dict:
    """Video background.

    Args:
        url: Video URL (YouTube, Vimeo, or hosted)
        fallback_image: Fallback image URL
        play_once: Play video only once
        play_on_mobile: Play video on mobile devices

    Returns:
        Background settings dict
    """
    settings = {
        "background_background": "video",
        "background_video_link": url,
    }

    if fallback_image:
        settings["background_video_fallback"] = {"url": fallback_image, "id": ""}

    if play_once:
        settings["background_play_once"] = "yes"

    if play_on_mobile:
        settings["background_play_on_mobile"] = "yes"

    return settings


# ============ Pre-built Section Templates ============

def section_hero(
    headline: str,
    subheadline: str = "",
    cta_text: str = "Get Started",
    cta_link: str = "#",
    cta_secondary_text: str = None,
    cta_secondary_link: str = None,
    background: dict = None,
    image_url: str = None,
    layout: str = "centered",  # centered, left, right (with image)
    min_height: int = 80,  # vh units
    headline_color: str = None,
    subheadline_color: str = None,
) -> dict:
    """Create a hero section.

    Args:
        headline: Main headline text
        subheadline: Secondary text below headline
        cta_text: Primary CTA button text
        cta_link: Primary CTA button link
        cta_secondary_text: Optional secondary CTA button text
        cta_secondary_link: Optional secondary CTA button link
        background: Background settings
        image_url: Image for left/right layouts
        layout: Section layout style
        min_height: Minimum height in vh
        headline_color: Headline text color
        subheadline_color: Subheadline text color

    Returns:
        Elementor section dict
    """
    # Default dark gradient background if none provided
    if background is None:
        background = background_gradient("#1a1a2e", "#16213e", 135)

    widgets = [
        widget_heading(
            text=headline,
            tag="h1",
            align="center" if layout == "centered" else "left",
            size="xxl",
            color=headline_color or "#ffffff",
        ),
    ]

    if subheadline:
        widgets.append(widget_spacer(20))
        widgets.append(widget_text(
            content=f"<p>{subheadline}</p>",
            align="center" if layout == "centered" else "left",
            color=subheadline_color or "#cccccc",
        ))

    widgets.append(widget_spacer(30))

    # CTA buttons
    if cta_secondary_text and cta_secondary_link:
        # Two buttons side by side - use inner section
        button1 = widget_button(
            text=cta_text,
            link=cta_link,
            style="success",
            size="lg",
            align="right" if layout == "centered" else "left",
        )
        button2 = widget_button(
            text=cta_secondary_text,
            link=cta_secondary_link,
            style="default",
            size="lg",
            align="left" if layout == "centered" else "left",
        )
        inner = create_inner_section(
            columns=[
                create_column([button1], width=50, vertical_align="middle", horizontal_align="end"),
                create_column([button2], width=50, vertical_align="middle", horizontal_align="start"),
            ],
            gap="narrow",
        )
        widgets.append(inner)
    else:
        widgets.append(widget_button(
            text=cta_text,
            link=cta_link,
            style="success",
            size="lg",
            align="center" if layout == "centered" else "left",
        ))

    if layout == "centered":
        columns = [create_column(
            widgets=widgets,
            width=100,
            vertical_align="middle",
        )]
    elif layout == "left" and image_url:
        columns = [
            create_column(
                widgets=widgets,
                width=50,
                vertical_align="middle",
            ),
            create_column(
                widgets=[widget_image(url=image_url, align="center")],
                width=50,
                vertical_align="middle",
            ),
        ]
    elif layout == "right" and image_url:
        columns = [
            create_column(
                widgets=[widget_image(url=image_url, align="center")],
                width=50,
                vertical_align="middle",
            ),
            create_column(
                widgets=widgets,
                width=50,
                vertical_align="middle",
            ),
        ]
    else:
        columns = [create_column(widgets=widgets, width=100, vertical_align="middle")]

    return create_section(
        columns=columns,
        layout="full_width",
        stretch=True,
        min_height={"size": min_height, "unit": "vh"},
        background=background,
        padding=_create_spacing(top=80, bottom=80),
    )


def section_features(
    features: list[dict],  # [{"icon": "fas fa-check", "title": "Feature", "description": "..."}]
    columns: int = 3,
    style: str = "icon-box",  # icon-box, cards
    headline: str = None,
    subheadline: str = None,
    icon_color: str = "#4361ee",
    background: dict = None,
) -> dict:
    """Create a features grid section.

    Args:
        features: List of features with icon, title, and description
        columns: Number of columns (2-4)
        style: Display style
        headline: Optional section headline
        subheadline: Optional section subheadline
        icon_color: Icon color for all features
        background: Section background

    Returns:
        Elementor section dict
    """
    # Header section if headline provided
    header_widgets = []
    if headline:
        header_widgets.append(widget_heading(
            text=headline,
            tag="h2",
            align="center",
            size="xl",
        ))
        if subheadline:
            header_widgets.append(widget_spacer(10))
            header_widgets.append(widget_text(
                content=f"<p>{subheadline}</p>",
                align="center",
            ))
        header_widgets.append(widget_spacer(40))

    # Create feature widgets
    feature_columns = []
    col_width = 100 // columns

    for feature in features:
        if style == "icon-box":
            w = widget_icon_box(
                title=feature.get("title", ""),
                description=feature.get("description", ""),
                icon=feature.get("icon", "fas fa-star"),
                icon_position="top",
                icon_color=icon_color,
                link=feature.get("link"),
            )
        else:
            # Card style - use text widgets with styling
            w = widget_icon_box(
                title=feature.get("title", ""),
                description=feature.get("description", ""),
                icon=feature.get("icon", "fas fa-star"),
                icon_position="top",
                icon_color=icon_color,
            )

        feature_columns.append(create_column(
            widgets=[w],
            width=col_width,
            padding=_create_spacing(top=20, right=20, bottom=20, left=20),
        ))

    # If we have header, wrap in structure
    if header_widgets:
        header_col = create_column(widgets=header_widgets, width=100)
        header_section = create_inner_section(columns=[header_col])

        # Add features as inner section
        features_inner = create_inner_section(columns=feature_columns, gap="wide")

        main_col = create_column(
            widgets=[header_section, features_inner],
            width=100,
        )
        cols = [main_col]
    else:
        cols = feature_columns

    section_bg = background or background_color("#f8f9fa")

    return create_section(
        columns=cols,
        layout="boxed",
        gap="wide",
        background=section_bg,
        padding=_create_spacing(top=80, bottom=80),
    )


def section_cta(
    headline: str,
    description: str = "",
    button_text: str = "Contact Us",
    button_link: str = "#",
    button_style: str = "success",
    background: dict = None,
    text_color: str = "#ffffff",
    align: str = "center",
) -> dict:
    """Create a call-to-action section.

    Args:
        headline: CTA headline
        description: CTA description text
        button_text: Button text
        button_link: Button link
        button_style: Button style
        background: Section background
        text_color: Text color
        align: Content alignment

    Returns:
        Elementor section dict
    """
    if background is None:
        background = background_gradient("#4361ee", "#3a0ca3", 135)

    widgets = [
        widget_heading(
            text=headline,
            tag="h2",
            align=align,
            size="xl",
            color=text_color,
        ),
    ]

    if description:
        widgets.append(widget_spacer(15))
        widgets.append(widget_text(
            content=f"<p>{description}</p>",
            align=align,
            color=text_color,
        ))

    widgets.append(widget_spacer(30))
    widgets.append(widget_button(
        text=button_text,
        link=button_link,
        style=button_style,
        size="lg",
        align=align,
    ))

    return create_section(
        columns=[create_column(widgets=widgets, width=100)],
        layout="full_width",
        stretch=True,
        background=background,
        padding=_create_spacing(top=80, bottom=80),
    )


def section_testimonials(
    testimonials: list[dict],  # [{"content": "...", "name": "John", "title": "CEO", "image": "..."}]
    headline: str = None,
    style: str = "cards",  # cards, simple
    columns: int = 2,
    background: dict = None,
) -> dict:
    """Create testimonials section.

    Args:
        testimonials: List of testimonials with content, name, title, image
        headline: Optional section headline
        style: Display style
        columns: Number of columns
        background: Section background

    Returns:
        Elementor section dict
    """
    header_widgets = []
    if headline:
        header_widgets.append(widget_heading(
            text=headline,
            tag="h2",
            align="center",
            size="xl",
        ))
        header_widgets.append(widget_spacer(40))

    testimonial_columns = []
    col_width = 100 // min(columns, len(testimonials))

    for test in testimonials:
        w = widget_testimonial(
            content=test.get("content", ""),
            name=test.get("name", ""),
            title=test.get("title", ""),
            image_url=test.get("image"),
            align="center",
        )
        testimonial_columns.append(create_column(
            widgets=[w],
            width=col_width,
            padding=_create_spacing(top=20, right=20, bottom=20, left=20),
        ))

    section_bg = background or background_color("#ffffff")

    if header_widgets:
        header_col = create_column(widgets=header_widgets, width=100)
        testimonials_inner = create_inner_section(columns=testimonial_columns, gap="wide")
        main_col = create_column(widgets=[
            create_inner_section(columns=[header_col]),
            testimonials_inner,
        ], width=100)
        cols = [main_col]
    else:
        cols = testimonial_columns

    return create_section(
        columns=cols,
        layout="boxed",
        gap="wide",
        background=section_bg,
        padding=_create_spacing(top=80, bottom=80),
    )


def section_pricing(
    plans: list[dict],  # [{"name": "Basic", "price": "$9", "period": "/mo", "features": [...], "cta_text": "Buy", "cta_link": "#"}]
    headline: str = None,
    subheadline: str = None,
    highlighted_plan: int = 1,  # Index of featured plan (0-based)
    background: dict = None,
) -> dict:
    """Create pricing table section.

    Args:
        plans: List of pricing plans
        headline: Optional section headline
        subheadline: Optional section subheadline
        highlighted_plan: Index of featured/highlighted plan
        background: Section background

    Returns:
        Elementor section dict
    """
    header_widgets = []
    if headline:
        header_widgets.append(widget_heading(
            text=headline,
            tag="h2",
            align="center",
            size="xl",
        ))
        if subheadline:
            header_widgets.append(widget_spacer(10))
            header_widgets.append(widget_text(
                content=f"<p>{subheadline}</p>",
                align="center",
            ))
        header_widgets.append(widget_spacer(40))

    plan_columns = []
    col_width = 100 // len(plans)

    for idx, plan in enumerate(plans):
        is_highlighted = idx == highlighted_plan

        widgets = []

        # Plan name
        widgets.append(widget_heading(
            text=plan.get("name", "Plan"),
            tag="h3",
            align="center",
            size="large",
        ))

        widgets.append(widget_spacer(15))

        # Price
        price_html = f'<p style="font-size: 48px; font-weight: bold;">{plan.get("price", "$0")}<span style="font-size: 16px;">{plan.get("period", "/mo")}</span></p>'
        widgets.append(widget_html(price_html))

        widgets.append(widget_spacer(20))

        # Features list
        if plan.get("features"):
            feature_items = [{"text": f, "icon": "fas fa-check"} for f in plan["features"]]
            widgets.append(widget_icon_list(
                items=feature_items,
                icon_color="#22c55e" if is_highlighted else "#4361ee",
            ))

        widgets.append(widget_spacer(25))

        # CTA button
        widgets.append(widget_button(
            text=plan.get("cta_text", "Get Started"),
            link=plan.get("cta_link", "#"),
            style="success" if is_highlighted else "default",
            size="lg",
            full_width=True,
        ))

        col_bg = background_color("#f0f9ff") if is_highlighted else None
        col_padding = _create_spacing(top=40, right=30, bottom=40, left=30)

        plan_columns.append(create_column(
            widgets=widgets,
            width=col_width,
            background=col_bg,
            padding=col_padding,
        ))

    section_bg = background or background_color("#ffffff")

    if header_widgets:
        header_col = create_column(widgets=header_widgets, width=100)
        plans_inner = create_inner_section(columns=plan_columns, gap="default")
        main_col = create_column(widgets=[
            create_inner_section(columns=[header_col]),
            plans_inner,
        ], width=100)
        cols = [main_col]
    else:
        cols = plan_columns

    return create_section(
        columns=cols,
        layout="boxed",
        gap="default",
        background=section_bg,
        padding=_create_spacing(top=80, bottom=80),
    )


def section_contact(
    headline: str = "Contact Us",
    description: str = None,
    show_map: bool = True,
    address: str = None,
    phone: str = None,
    email: str = None,
    form_fields: list[str] = None,  # ["name", "email", "phone", "message"]
    form_email_to: str = None,
    background: dict = None,
) -> dict:
    """Create contact section with form and info.

    Args:
        headline: Section headline
        description: Optional description
        show_map: Whether to show Google Map
        address: Business address
        phone: Contact phone number
        email: Contact email address
        form_fields: List of form field names to include
        form_email_to: Email address for form submissions
        background: Section background

    Returns:
        Elementor section dict
    """
    if form_fields is None:
        form_fields = ["name", "email", "message"]

    # Left column - contact info
    info_widgets = [
        widget_heading(text=headline, tag="h2", align="left", size="xl"),
    ]

    if description:
        info_widgets.append(widget_spacer(15))
        info_widgets.append(widget_text(content=f"<p>{description}</p>", align="left"))

    info_widgets.append(widget_spacer(30))

    # Contact details
    contact_items = []
    if address:
        contact_items.append({"text": address, "icon": "fas fa-map-marker-alt"})
    if phone:
        contact_items.append({"text": phone, "icon": "fas fa-phone"})
    if email:
        contact_items.append({"text": email, "icon": "fas fa-envelope"})

    if contact_items:
        info_widgets.append(widget_icon_list(
            items=contact_items,
            icon_color="#4361ee",
            space_between=15,
        ))

    # Map
    if show_map and address:
        info_widgets.append(widget_spacer(30))
        info_widgets.append(widget_google_maps(
            address=address,
            zoom=14,
            height=200,
        ))

    # Right column - form
    fields = []
    field_map = {
        "name": {"type": "text", "label": "Name", "required": True, "placeholder": "Your name"},
        "email": {"type": "email", "label": "Email", "required": True, "placeholder": "Your email"},
        "phone": {"type": "tel", "label": "Phone", "required": False, "placeholder": "Your phone"},
        "subject": {"type": "text", "label": "Subject", "required": False, "placeholder": "Subject"},
        "message": {"type": "textarea", "label": "Message", "required": True, "placeholder": "Your message", "rows": 5},
    }

    for field_name in form_fields:
        if field_name in field_map:
            fields.append(field_map[field_name])

    form_widgets = [
        widget_form(
            fields=fields,
            submit_text="Send Message",
            email_to=form_email_to,
            button_size="lg",
            button_width="100",
        ),
    ]

    section_bg = background or background_color("#f8f9fa")

    return create_section(
        columns=[
            create_column(widgets=info_widgets, width=50),
            create_column(widgets=form_widgets, width=50),
        ],
        layout="boxed",
        gap="wide",
        background=section_bg,
        padding=_create_spacing(top=80, bottom=80),
    )


def section_stats(
    stats: list[dict],  # [{"number": 100, "label": "Clients", "prefix": "", "suffix": "+"}]
    headline: str = None,
    background: dict = None,
    number_color: str = "#ffffff",
    label_color: str = "#cccccc",
) -> dict:
    """Create statistics/counter section.

    Args:
        stats: List of statistics with number, label, prefix, suffix
        headline: Optional section headline
        background: Section background
        number_color: Number text color
        label_color: Label text color

    Returns:
        Elementor section dict
    """
    if background is None:
        background = background_gradient("#1a1a2e", "#16213e", 135)

    header_widgets = []
    if headline:
        header_widgets.append(widget_heading(
            text=headline,
            tag="h2",
            align="center",
            size="xl",
            color=number_color,
        ))
        header_widgets.append(widget_spacer(40))

    stat_columns = []
    col_width = 100 // len(stats)

    for stat in stats:
        w = widget_counter(
            number=stat.get("number", 0),
            title=stat.get("label", ""),
            prefix=stat.get("prefix", ""),
            suffix=stat.get("suffix", ""),
            align="center",
            number_color=number_color,
            title_color=label_color,
        )
        stat_columns.append(create_column(
            widgets=[w],
            width=col_width,
        ))

    if header_widgets:
        header_col = create_column(widgets=header_widgets, width=100)
        stats_inner = create_inner_section(columns=stat_columns, gap="wide")
        main_col = create_column(widgets=[
            create_inner_section(columns=[header_col]),
            stats_inner,
        ], width=100)
        cols = [main_col]
    else:
        cols = stat_columns

    return create_section(
        columns=cols,
        layout="full_width",
        stretch=True,
        background=background,
        padding=_create_spacing(top=80, bottom=80),
    )


def section_team(
    members: list[dict],  # [{"name": "John", "title": "CEO", "image": "...", "bio": "...", "social": {"twitter": "...", "linkedin": "..."}}]
    headline: str = None,
    subheadline: str = None,
    columns: int = 4,
    background: dict = None,
) -> dict:
    """Create team members section.

    Args:
        members: List of team members
        headline: Optional section headline
        subheadline: Optional section subheadline
        columns: Number of columns
        background: Section background

    Returns:
        Elementor section dict
    """
    header_widgets = []
    if headline:
        header_widgets.append(widget_heading(
            text=headline,
            tag="h2",
            align="center",
            size="xl",
        ))
        if subheadline:
            header_widgets.append(widget_spacer(10))
            header_widgets.append(widget_text(
                content=f"<p>{subheadline}</p>",
                align="center",
            ))
        header_widgets.append(widget_spacer(40))

    member_columns = []
    col_width = 100 // min(columns, len(members))

    for member in members:
        widgets = []

        # Member image
        if member.get("image"):
            widgets.append(widget_image(
                url=member["image"],
                align="center",
                size="medium",
                border_radius=_create_spacing(top=100, right=100, bottom=100, left=100, unit="%"),
            ))
            widgets.append(widget_spacer(20))

        # Name
        widgets.append(widget_heading(
            text=member.get("name", ""),
            tag="h4",
            align="center",
        ))

        # Title
        if member.get("title"):
            widgets.append(widget_text(
                content=f"<p>{member['title']}</p>",
                align="center",
                color="#666666",
            ))

        # Bio
        if member.get("bio"):
            widgets.append(widget_spacer(10))
            widgets.append(widget_text(
                content=f"<p>{member['bio']}</p>",
                align="center",
            ))

        # Social icons
        if member.get("social"):
            widgets.append(widget_spacer(15))
            social_icons = []
            for platform, link in member["social"].items():
                social_icons.append({"platform": platform, "link": link})
            widgets.append(widget_social_icons(
                icons=social_icons,
                shape="circle",
                align="center",
            ))

        member_columns.append(create_column(
            widgets=widgets,
            width=col_width,
            padding=_create_spacing(top=20, right=20, bottom=20, left=20),
        ))

    section_bg = background or background_color("#ffffff")

    if header_widgets:
        header_col = create_column(widgets=header_widgets, width=100)
        members_inner = create_inner_section(columns=member_columns, gap="default")
        main_col = create_column(widgets=[
            create_inner_section(columns=[header_col]),
            members_inner,
        ], width=100)
        cols = [main_col]
    else:
        cols = member_columns

    return create_section(
        columns=cols,
        layout="boxed",
        gap="default",
        background=section_bg,
        padding=_create_spacing(top=80, bottom=80),
    )


def section_faq(
    questions: list[dict],  # [{"question": "...", "answer": "..."}]
    headline: str = "Frequently Asked Questions",
    subheadline: str = None,
    background: dict = None,
) -> dict:
    """Create FAQ accordion section.

    Args:
        questions: List of FAQ items with question and answer
        headline: Section headline
        subheadline: Optional section subheadline
        background: Section background

    Returns:
        Elementor section dict
    """
    widgets = []

    if headline:
        widgets.append(widget_heading(
            text=headline,
            tag="h2",
            align="center",
            size="xl",
        ))
        if subheadline:
            widgets.append(widget_spacer(10))
            widgets.append(widget_text(
                content=f"<p>{subheadline}</p>",
                align="center",
            ))
        widgets.append(widget_spacer(40))

    # Convert questions to accordion items
    accordion_items = [
        {"title": q["question"], "content": q["answer"]}
        for q in questions
    ]

    widgets.append(widget_accordion(
        items=accordion_items,
        icon="fas fa-plus",
        active_icon="fas fa-minus",
    ))

    section_bg = background or background_color("#f8f9fa")

    return create_section(
        columns=[create_column(widgets=widgets, width=100)],
        layout="boxed",
        background=section_bg,
        padding=_create_spacing(top=80, bottom=80),
    )


def section_logos(
    logos: list[str],  # List of image URLs
    headline: str = "Trusted By",
    grayscale: bool = True,
    background: dict = None,
    columns: int = 6,
) -> dict:
    """Create logo carousel/grid section.

    Args:
        logos: List of logo image URLs
        headline: Optional section headline
        grayscale: Whether to display logos in grayscale
        background: Section background
        columns: Number of columns for logo grid

    Returns:
        Elementor section dict
    """
    widgets = []

    if headline:
        widgets.append(widget_heading(
            text=headline,
            tag="h3",
            align="center",
            size="medium",
            color="#666666",
        ))
        widgets.append(widget_spacer(30))

    # Create logo images
    logo_images = [{"url": url, "alt": "Partner logo"} for url in logos]

    # Use image carousel for logos
    widgets.append(widget_image_carousel(
        images=logo_images,
        slides_to_show=min(columns, len(logos)),
        slides_to_scroll=1,
        navigation="none" if len(logos) <= columns else "dots",
        autoplay=True,
        autoplay_speed=3000,
        image_stretch="no",
    ))

    section_bg = background or background_color("#ffffff")

    # Add grayscale CSS if needed
    css_class = "logo-grayscale" if grayscale else ""

    return create_section(
        columns=[create_column(widgets=widgets, width=100)],
        layout="boxed",
        background=section_bg,
        padding=_create_spacing(top=60, bottom=60),
        css_classes=css_class,
    )


def section_two_column_content(
    left_content: list[dict],  # List of widgets
    right_content: list[dict],  # List of widgets
    left_width: int = 50,
    vertical_align: str = "middle",
    background: dict = None,
    reverse_on_mobile: bool = False,
) -> dict:
    """Create a flexible two-column content section.

    Args:
        left_content: Widgets for left column
        right_content: Widgets for right column
        left_width: Width of left column (right gets remainder)
        vertical_align: Vertical alignment of content
        background: Section background
        reverse_on_mobile: Reverse column order on mobile

    Returns:
        Elementor section dict
    """
    right_width = 100 - left_width

    settings = {}
    if reverse_on_mobile:
        settings["reverse_order_mobile"] = "yes"

    section_bg = background or background_color("#ffffff")

    return create_section(
        columns=[
            create_column(widgets=left_content, width=left_width, vertical_align=vertical_align),
            create_column(widgets=right_content, width=right_width, vertical_align=vertical_align),
        ],
        layout="boxed",
        gap="wide",
        background=section_bg,
        padding=_create_spacing(top=80, bottom=80),
    )


# ============ Page Templates ============

def page_landing(
    hero: dict,
    sections: list[dict],
    cta: dict = None,
) -> list[dict]:
    """Assemble a complete landing page from sections.

    Args:
        hero: Hero section dict
        sections: List of content section dicts
        cta: Optional final CTA section dict

    Returns:
        List of Elementor sections for the full page
    """
    page = [hero]
    page.extend(sections)
    if cta:
        page.append(cta)
    return page


def page_about(
    company_name: str,
    story: str,
    mission: str = None,
    vision: str = None,
    values: list[dict] = None,  # [{"icon": "fas fa-heart", "title": "Integrity", "description": "..."}]
    team: list[dict] = None,
    stats: list[dict] = None,
    timeline: list[dict] = None,  # [{"year": "2020", "title": "Founded", "description": "..."}]
) -> list[dict]:
    """Generate an About Us page.

    Args:
        company_name: Company/organization name
        story: Company story/history text
        mission: Mission statement
        vision: Vision statement
        values: List of company values
        team: List of team members
        stats: List of statistics
        timeline: List of timeline events

    Returns:
        List of Elementor sections for About page
    """
    sections = []

    # Hero section
    sections.append(section_hero(
        headline=f"About {company_name}",
        subheadline="Our story, mission, and the people behind our success.",
        cta_text="Meet the Team",
        cta_link="#team",
        layout="centered",
        min_height=50,
    ))

    # Story section
    story_widgets = [
        widget_heading(text="Our Story", tag="h2", align="left", size="xl"),
        widget_spacer(20),
        widget_text(content=f"<p>{story}</p>", align="left"),
    ]

    sections.append(create_section(
        columns=[create_column(widgets=story_widgets, width=100)],
        layout="boxed",
        padding=_create_spacing(top=80, bottom=80),
    ))

    # Mission & Vision
    if mission or vision:
        mv_columns = []
        col_width = 50 if (mission and vision) else 100

        if mission:
            mission_widgets = [
                widget_icon_box(
                    title="Our Mission",
                    description=mission,
                    icon="fas fa-bullseye",
                    icon_position="top",
                    icon_color="#4361ee",
                ),
            ]
            mv_columns.append(create_column(widgets=mission_widgets, width=col_width))

        if vision:
            vision_widgets = [
                widget_icon_box(
                    title="Our Vision",
                    description=vision,
                    icon="fas fa-eye",
                    icon_position="top",
                    icon_color="#4361ee",
                ),
            ]
            mv_columns.append(create_column(widgets=vision_widgets, width=col_width))

        sections.append(create_section(
            columns=mv_columns,
            layout="boxed",
            gap="wide",
            background=background_color("#f8f9fa"),
            padding=_create_spacing(top=80, bottom=80),
        ))

    # Values
    if values:
        sections.append(section_features(
            features=values,
            columns=min(4, len(values)),
            headline="Our Values",
            subheadline="The principles that guide everything we do.",
        ))

    # Stats
    if stats:
        sections.append(section_stats(
            stats=stats,
            headline="By the Numbers",
        ))

    # Team
    if team:
        sections.append(section_team(
            members=team,
            headline="Meet Our Team",
            subheadline="The talented people behind our success.",
            columns=min(4, len(team)),
        ))

    return sections


def page_services(
    headline: str,
    services: list[dict],  # [{"title": "...", "description": "...", "icon": "...", "features": [...], "link": "..."}]
    intro: str = None,
    cta: dict = None,  # {"headline": "...", "description": "...", "button_text": "...", "button_link": "..."}
) -> list[dict]:
    """Generate a Services page.

    Args:
        headline: Page headline
        intro: Optional introduction text
        services: List of services with details
        cta: Optional CTA section config

    Returns:
        List of Elementor sections for Services page
    """
    sections = []

    # Hero
    sections.append(section_hero(
        headline=headline,
        subheadline=intro or "Discover how we can help you achieve your goals.",
        cta_text="Get Started",
        cta_link="#contact",
        layout="centered",
        min_height=50,
    ))

    # Services overview
    service_features = [
        {
            "icon": s.get("icon", "fas fa-cog"),
            "title": s.get("title", ""),
            "description": s.get("description", "")[:150] + "..." if len(s.get("description", "")) > 150 else s.get("description", ""),
            "link": s.get("link"),
        }
        for s in services
    ]

    sections.append(section_features(
        features=service_features,
        columns=min(3, len(services)),
        headline="Our Services",
    ))

    # Detailed service sections
    for idx, service in enumerate(services):
        is_even = idx % 2 == 0

        # Service detail widgets
        detail_widgets = [
            widget_heading(text=service.get("title", ""), tag="h2", align="left", size="xl"),
            widget_spacer(15),
            widget_text(content=f"<p>{service.get('description', '')}</p>", align="left"),
        ]

        # Features list if provided
        if service.get("features"):
            detail_widgets.append(widget_spacer(20))
            feature_items = [{"text": f, "icon": "fas fa-check"} for f in service["features"]]
            detail_widgets.append(widget_icon_list(items=feature_items, icon_color="#22c55e"))

        # Link button if provided
        if service.get("link"):
            detail_widgets.append(widget_spacer(25))
            detail_widgets.append(widget_button(
                text="Learn More",
                link=service["link"],
                align="left",
            ))

        # Image placeholder
        image_widgets = []
        if service.get("image"):
            image_widgets.append(widget_image(url=service["image"], align="center"))
        else:
            image_widgets.append(widget_icon_box(
                title="",
                description="",
                icon=service.get("icon", "fas fa-cog"),
                icon_color="#4361ee",
                icon_size={"size": 100, "unit": "px"},
            ))

        bg = background_color("#f8f9fa") if is_even else background_color("#ffffff")

        if is_even:
            cols = [
                create_column(widgets=detail_widgets, width=50, vertical_align="middle"),
                create_column(widgets=image_widgets, width=50, vertical_align="middle"),
            ]
        else:
            cols = [
                create_column(widgets=image_widgets, width=50, vertical_align="middle"),
                create_column(widgets=detail_widgets, width=50, vertical_align="middle"),
            ]

        sections.append(create_section(
            columns=cols,
            layout="boxed",
            gap="wide",
            background=bg,
            padding=_create_spacing(top=80, bottom=80),
        ))

    # CTA
    if cta:
        sections.append(section_cta(
            headline=cta.get("headline", "Ready to Get Started?"),
            description=cta.get("description", ""),
            button_text=cta.get("button_text", "Contact Us"),
            button_link=cta.get("button_link", "#contact"),
        ))

    return sections


def page_contact(
    headline: str = "Get In Touch",
    description: str = None,
    address: str = None,
    phone: str = None,
    email: str = None,
    map_address: str = None,
    form_email_to: str = None,
    social_links: dict = None,  # {"facebook": "...", "twitter": "...", etc.}
    office_hours: str = None,
) -> list[dict]:
    """Generate a Contact page.

    Args:
        headline: Page headline
        description: Optional description text
        address: Business address
        phone: Contact phone number
        email: Contact email address
        map_address: Address for Google Maps (defaults to address)
        form_email_to: Email for form submissions
        social_links: Social media links
        office_hours: Office hours text

    Returns:
        List of Elementor sections for Contact page
    """
    sections = []

    # Hero section
    sections.append(section_hero(
        headline=headline,
        subheadline=description or "We'd love to hear from you. Send us a message and we'll respond as soon as possible.",
        cta_text="Send Message",
        cta_link="#contact-form",
        layout="centered",
        min_height=40,
    ))

    # Main contact section
    sections.append(section_contact(
        headline="Send Us a Message",
        description="Fill out the form below and we'll get back to you within 24 hours.",
        show_map=bool(map_address or address),
        address=address,
        phone=phone,
        email=email,
        form_fields=["name", "email", "phone", "subject", "message"],
        form_email_to=form_email_to,
    ))

    # Additional info section
    info_widgets = []

    if office_hours:
        info_widgets.append(widget_icon_box(
            title="Office Hours",
            description=office_hours,
            icon="fas fa-clock",
            icon_position="left",
            icon_color="#4361ee",
        ))

    if social_links:
        social_icons = [{"platform": k, "link": v} for k, v in social_links.items()]
        info_widgets.append(widget_spacer(30))
        info_widgets.append(widget_heading(text="Follow Us", tag="h4", align="left"))
        info_widgets.append(widget_spacer(15))
        info_widgets.append(widget_social_icons(icons=social_icons, shape="circle", align="left"))

    if info_widgets:
        sections.append(create_section(
            columns=[create_column(widgets=info_widgets, width=100)],
            layout="boxed",
            background=background_color("#f8f9fa"),
            padding=_create_spacing(top=60, bottom=60),
        ))

    return sections


def page_portfolio(
    projects: list[dict],  # [{"title": "...", "category": "...", "image": "...", "description": "...", "link": "..."}]
    headline: str = "Our Work",
    intro: str = None,
    categories: list[str] = None,  # For filtering
    columns: int = 3,
) -> list[dict]:
    """Generate a Portfolio page.

    Args:
        headline: Page headline
        intro: Optional introduction text
        projects: List of portfolio projects
        categories: List of category filters
        columns: Number of columns for project grid

    Returns:
        List of Elementor sections for Portfolio page
    """
    sections = []

    # Hero
    sections.append(section_hero(
        headline=headline,
        subheadline=intro or "Explore our latest projects and see what we can do for you.",
        cta_text="Start a Project",
        cta_link="#contact",
        layout="centered",
        min_height=40,
    ))

    # Projects grid
    project_columns = []
    col_width = 100 // columns

    for project in projects:
        widgets = []

        # Project image
        if project.get("image"):
            widgets.append(widget_image(
                url=project["image"],
                align="center",
                link=project.get("link"),
                hover_animation="grow",
            ))

        widgets.append(widget_spacer(15))

        # Category badge
        if project.get("category"):
            widgets.append(widget_text(
                content=f'<p style="color: #4361ee; font-size: 12px; text-transform: uppercase; letter-spacing: 1px;">{project["category"]}</p>',
                align="center",
            ))

        # Title
        widgets.append(widget_heading(
            text=project.get("title", ""),
            tag="h4",
            align="center",
            link=project.get("link"),
        ))

        # Description
        if project.get("description"):
            widgets.append(widget_text(
                content=f"<p>{project['description'][:100]}...</p>",
                align="center",
            ))

        # View link
        if project.get("link"):
            widgets.append(widget_spacer(10))
            widgets.append(widget_button(
                text="View Project",
                link=project["link"],
                style="default",
                size="sm",
                align="center",
            ))

        project_columns.append(create_column(
            widgets=widgets,
            width=col_width,
            padding=_create_spacing(top=20, right=20, bottom=30, left=20),
        ))

    # Break into rows if needed
    row_sections = []
    for i in range(0, len(project_columns), columns):
        row = project_columns[i:i + columns]
        # Pad last row if needed
        while len(row) < columns:
            row.append(create_column(widgets=[], width=col_width))
        row_sections.append(create_inner_section(columns=row, gap="default"))

    portfolio_widgets = []
    for row in row_sections:
        portfolio_widgets.append(row)

    sections.append(create_section(
        columns=[create_column(widgets=portfolio_widgets, width=100)],
        layout="boxed",
        padding=_create_spacing(top=60, bottom=60),
    ))

    return sections


# ============ Utility Functions ============

def to_elementor_json(sections: list[dict]) -> str:
    """Convert sections list to Elementor JSON string.

    Args:
        sections: List of Elementor section dicts

    Returns:
        JSON string ready for Elementor import
    """
    return json.dumps(sections)


def from_elementor_json(json_string: str) -> list[dict]:
    """Parse Elementor JSON string to sections list.

    Args:
        json_string: Elementor JSON data

    Returns:
        List of Elementor section dicts
    """
    return json.loads(json_string)


def apply_color_scheme(
    sections: list[dict],
    primary: str,
    secondary: str,
    accent: str,
    text: str = "#333333",
    text_light: str = "#666666",
    background: str = "#ffffff",
    background_alt: str = "#f8f9fa",
) -> list[dict]:
    """Apply a color scheme to all sections.

    Args:
        sections: List of Elementor section dicts
        primary: Primary brand color
        secondary: Secondary brand color
        accent: Accent color (for CTAs, highlights)
        text: Main text color
        text_light: Light text color
        background: Main background color
        background_alt: Alternate background color

    Returns:
        Sections with updated color scheme
    """
    # Deep copy to avoid modifying original
    sections = copy.deepcopy(sections)

    color_map = {
        "#4361ee": primary,  # Default primary
        "#3a0ca3": secondary,  # Default secondary
        "#22c55e": accent,  # Default accent
        "#333333": text,
        "#666666": text_light,
        "#ffffff": background,
        "#f8f9fa": background_alt,
    }

    def replace_colors(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: replace_colors(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [replace_colors(item) for item in obj]
        elif isinstance(obj, str) and obj.lower() in [c.lower() for c in color_map.keys()]:
            # Find the matching color (case-insensitive)
            for old_color, new_color in color_map.items():
                if obj.lower() == old_color.lower():
                    return new_color
        return obj

    return replace_colors(sections)


def regenerate_ids(sections: list[dict]) -> list[dict]:
    """Regenerate all element IDs (useful for duplicating layouts).

    Args:
        sections: List of Elementor section dicts

    Returns:
        Sections with new unique IDs
    """
    sections = copy.deepcopy(sections)

    def regenerate(obj: Any) -> Any:
        if isinstance(obj, dict):
            if "id" in obj and isinstance(obj["id"], str) and len(obj["id"]) == 7:
                obj["id"] = _generate_id()
            if "_id" in obj and isinstance(obj["_id"], str):
                obj["_id"] = _generate_id()[:8]
            for key, value in obj.items():
                obj[key] = regenerate(value)
        elif isinstance(obj, list):
            return [regenerate(item) for item in obj]
        return obj

    return regenerate(sections)


def merge_sections(*section_lists: list[dict]) -> list[dict]:
    """Merge multiple section lists into one page.

    Args:
        *section_lists: Variable number of section lists to merge

    Returns:
        Combined list of all sections
    """
    result = []
    for sections in section_lists:
        result.extend(sections)
    return result


def create_responsive_settings(
    desktop: Any,
    tablet: Any = None,
    mobile: Any = None,
) -> dict:
    """Create responsive settings object.

    Args:
        desktop: Desktop value
        tablet: Tablet value (optional)
        mobile: Mobile value (optional)

    Returns:
        Responsive settings dict
    """
    settings = {"desktop": desktop}
    if tablet is not None:
        settings["tablet"] = tablet
    if mobile is not None:
        settings["mobile"] = mobile
    return settings


def export_template(
    sections: list[dict],
    title: str,
    template_type: str = "page",  # page, section, popup
) -> dict:
    """Export sections as Elementor template format.

    Args:
        sections: List of Elementor section dicts
        title: Template title
        template_type: Type of template

    Returns:
        Elementor template export format dict
    """
    return {
        "version": "0.4",
        "title": title,
        "type": template_type,
        "content": sections,
    }


def import_template(template_data: dict) -> list[dict]:
    """Import Elementor template and extract sections.

    Args:
        template_data: Elementor template dict

    Returns:
        List of Elementor section dicts
    """
    return template_data.get("content", [])


# ============ Widget Helpers ============

def wrap_in_container(
    widgets: list[dict],
    max_width: int = 1140,
    align: str = "center",
    padding: dict = None,
) -> dict:
    """Wrap widgets in a centered container.

    Args:
        widgets: List of widgets to wrap
        max_width: Maximum container width in pixels
        align: Container alignment
        padding: Optional padding

    Returns:
        Section with contained widgets
    """
    col_padding = padding or _create_spacing(top=0, right=20, bottom=0, left=20)

    return create_section(
        columns=[create_column(
            widgets=widgets,
            width=100,
            padding=col_padding,
        )],
        layout="boxed",
        content_width="boxed",
    )
