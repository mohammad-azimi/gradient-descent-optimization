import argparse
from copy import deepcopy
import json
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION_START
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from lxml import etree


ROOT = Path(__file__).resolve().parent
ASSETS = ROOT/"polytech_assets"
VML = "urn:schemas-microsoft-com:vml"
MARKER = "polytech-gradient-descent-theme-v1"


def set_font(run, size=None, bold=None, color="000000"):
    run.font.name = "Arial"
    run.font.color.rgb = RGBColor.from_string(color)
    properties = run._r.get_or_add_rPr()
    fonts = properties.find(qn("w:rFonts"))
    if fonts is not None:
        for key in ["asciiTheme", "hAnsiTheme", "eastAsiaTheme", "cstheme"]:
            fonts.attrib.pop(qn("w:"+key), None)
        for key in ["ascii", "hAnsi", "eastAsia", "cs"]:
            fonts.set(qn("w:"+key), "Arial")
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold


def crop_picture(shape, roi, source_size):
    x0, y0, x1, y1 = roi
    width, height = source_size
    crop = OxmlElement("a:srcRect")
    for name, value in [("l", x0/width), ("t", y0/height),
                        ("r", 1-x1/width), ("b", 1-y1/height)]:
        crop.set(name, str(round(value*100000)))
    fill = shape._inline.graphic.graphicData.pic.blipFill
    for node in list(fill):
        if node.tag == qn("a:srcRect"):
            fill.remove(node)
    stretch = fill.find(qn("a:stretch"))
    fill.insert(list(fill).index(stretch) if stretch is not None else 1, crop)


def anchored_picture(paragraph, file, x, y, width, height, roi=None, source_size=None, picture_id=50000):
    shape = paragraph.add_run().add_picture(str(file), width=Inches(width), height=Inches(height))
    if roi is not None:
        crop_picture(shape, roi, source_size)
    inline = shape._inline
    anchor = OxmlElement("wp:anchor")
    for key, value in {"distT": "0", "distB": "0", "distL": "0", "distR": "0",
                       "simplePos": "0", "relativeHeight": str(250000000+picture_id),
                       "behindDoc": "0", "locked": "0", "layoutInCell": "1", "allowOverlap": "1"}.items():
        anchor.set(key, value)
    simple = OxmlElement("wp:simplePos")
    simple.set("x", "0")
    simple.set("y", "0")
    anchor.append(simple)
    for tag, value in [("wp:positionH", x), ("wp:positionV", y)]:
        position = OxmlElement(tag)
        position.set("relativeFrom", "page")
        offset = OxmlElement("wp:posOffset")
        offset.text = str(int(Inches(value)))
        position.append(offset)
        anchor.append(position)
    anchor.append(deepcopy(inline.extent))
    effect = OxmlElement("wp:effectExtent")
    for key in ["l", "t", "r", "b"]:
        effect.set(key, "0")
    anchor.append(effect)
    anchor.append(OxmlElement("wp:wrapNone"))
    doc_pr = deepcopy(inline.docPr)
    doc_pr.set("id", str(picture_id))
    doc_pr.set("name", "Polytech artwork "+str(picture_id))
    doc_pr.set("descr", "Polytech artwork from the supplied report reference")
    anchor.append(doc_pr)
    properties = inline.find(qn("wp:cNvGraphicFramePr"))
    if properties is not None:
        anchor.append(deepcopy(properties))
    anchor.append(deepcopy(inline.graphic))
    inline.getparent().replace(inline, anchor)


def text_box(paragraph, lines, x, y, width, height, font_size, shape_id):
    text_height = font_size*1.1*len(lines)/72.0
    y += max(0.0, (height-text_height)/2.0)
    height = text_height+0.04
    pict = OxmlElement("w:pict")
    shape = etree.SubElement(pict, "{"+VML+"}rect", nsmap={"v": VML})
    shape.set("id", shape_id)
    shape.set("filled", "f")
    shape.set("stroked", "f")
    shape.set("style", ";".join([
        "position:absolute", f"margin-left:{x*72:.3f}pt", f"margin-top:{y*72:.3f}pt",
        f"width:{width*72:.3f}pt", f"height:{height*72:.3f}pt", "z-index:251659999",
        "mso-position-horizontal-relative:page", "mso-position-vertical-relative:page",
        "v-text-anchor:middle",
    ]))
    textbox = etree.SubElement(shape, "{"+VML+"}textbox")
    textbox.set("inset", "7pt,0pt,7pt,0pt")
    content = OxmlElement("w:txbxContent")
    for text in lines:
        p = OxmlElement("w:p")
        properties = OxmlElement("w:pPr")
        align = OxmlElement("w:jc")
        align.set(qn("w:val"), "center")
        properties.append(align)
        spacing = OxmlElement("w:spacing")
        for key, value in [("before", "0"), ("after", "0"), ("line", "264"), ("lineRule", "auto")]:
            spacing.set(qn("w:"+key), value)
        properties.append(spacing)
        p.append(properties)
        run = OxmlElement("w:r")
        pr = OxmlElement("w:rPr")
        fonts = OxmlElement("w:rFonts")
        fonts.set(qn("w:ascii"), "Arial")
        fonts.set(qn("w:hAnsi"), "Arial")
        pr.append(fonts)
        pr.append(OxmlElement("w:b"))
        color = OxmlElement("w:color")
        color.set(qn("w:val"), "FFFFFF")
        pr.append(color)
        size = OxmlElement("w:sz")
        size.set(qn("w:val"), str(round(font_size*2)))
        pr.append(size)
        run.append(pr)
        node = OxmlElement("w:t")
        node.text = text
        run.append(node)
        p.append(run)
        content.append(p)
    textbox.append(content)
    paragraph.add_run()._r.append(pict)


def clear_story(story):
    for node in list(story._element):
        story._element.remove(node)
    story._element.append(OxmlElement("w:p"))


def apply_styles(doc):
    for node in doc.styles.element.xpath(".//w:pBdr"):
        node.getparent().remove(node)
    for name, size in [("Normal", 10.5), ("Title", 22), ("Subtitle", 14),
                       ("Heading 1", 16), ("Heading 2", 12), ("Heading 3", 11), ("Caption", 9)]:
        style = doc.styles[name]
        style.font.name = "Arial"
        style.font.size = Pt(size)
        style.font.color.rgb = RGBColor(0, 0, 0)
        style.font.bold = name in ["Title", "Heading 1", "Heading 2", "Heading 3"]
    normal = doc.styles["Normal"].paragraph_format
    normal.space_after = Pt(7)
    normal.line_spacing = 1.12
    for name, before, after in [("Heading 1", 0, 12), ("Heading 2", 10, 6), ("Heading 3", 10, 5)]:
        pf = doc.styles[name].paragraph_format
        pf.space_before, pf.space_after = Pt(before), Pt(after)
        pf.keep_with_next = True
    title = doc.styles["Title"].paragraph_format
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.line_spacing = 1.0
    title.space_after = Pt(14)
    caption = doc.styles["Caption"].paragraph_format
    caption.line_spacing = 1.0
    caption.space_after = Pt(9)
    for paragraph in doc.paragraphs:
        for run in paragraph.runs:
            set_font(run)
            if paragraph.style.name == "Normal" and run.text:
                run.font.size = Pt(10.5)
            elif paragraph.style.name == "Caption":
                run.bold = False
    for table in doc.tables:
        for i, row in enumerate(table.rows):
            for cell in row.cells:
                cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
                properties = cell._tc.get_or_add_tcPr()
                for node in list(properties):
                    if node.tag in [qn("w:shd"), qn("w:tcBorders")]:
                        properties.remove(node)
                shading = OxmlElement("w:shd")
                shading.set(qn("w:fill"), "E7E7E7" if i == 0 else "FFFFFF")
                properties.append(shading)
                borders = OxmlElement("w:tcBorders")
                for side in ["top", "left", "bottom", "right"]:
                    border = OxmlElement("w:"+side)
                    for key, value in [("val", "single"), ("sz", "4"), ("color", "D9D9D9")]:
                        border.set(qn("w:"+key), value)
                    borders.append(border)
                properties.append(borders)
                for paragraph in cell.paragraphs:
                    paragraph.paragraph_format.space_after = Pt(0)
                    paragraph.paragraph_format.line_spacing = 1.05
                    for run in paragraph.runs:
                        set_font(run, size=9, bold=i == 0)


def add_cover(doc, data, assets):
    prefix = []

    def paragraph(text, size, bold=False, before=0, after=0, style=None):
        p = doc.add_paragraph(style=style)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(before)
        p.paragraph_format.space_after = Pt(after)
        p.paragraph_format.line_spacing = 1.1
        if text:
            set_font(p.add_run(text), size=size, bold=bold)
        prefix.append(p._p)
        return p

    hero = paragraph("", 1)
    hero.alignment = WD_ALIGN_PARAGRAPH.LEFT
    hero.paragraph_format.left_indent = Inches(-0.65)
    hero.paragraph_format.right_indent = Inches(-0.65)
    hero.paragraph_format.line_spacing = 1.0
    shape = hero.add_run().add_picture(str(assets/"cover_source.png"), width=Inches(8.5), height=Inches(4.78125))
    for key in ["distT", "distB", "distL", "distR"]:
        shape._inline.set(key, "0")
    effect = OxmlElement("wp:effectExtent")
    for key in ["l", "t", "r", "b"]:
        effect.set(key, "0")
    shape._inline.insert(1, effect)
    shape._inline.docPr.set("descr", "Polytech cover artwork from the supplied reference")
    scale = 8.5/2400
    blue_top = 403*scale
    for index, source_height in enumerate([300, 300, 300, 47]):
        anchored_picture(hero, assets/"cover_source.png", 0, blue_top+index*300*scale,
                         1050*scale+0.01, source_height*scale+0.012,
                         roi=(0, 410, 1050, 410+source_height), source_size=(2400, 1350),
                         picture_id=50100+index)
    text_box(hero, data["cover_lines"], 0, blue_top, 1050*scale, 947*scale, 22, "PolytechCoverTitle")
    paragraph(data["report_category"], 11, bold=True, before=18, after=8)
    paragraph(data["report_title"], 22, bold=True, after=15, style="Title")
    paragraph(data["institute"], 11.5, after=18)
    for author in data["authors"]:
        paragraph(author, 18, bold=True)
    paragraph(data["group"]+"   "+data["subgroup"], 11, after=13)
    paragraph(data["course"], 12, after=9)
    p = paragraph("", 11.5)
    set_font(p.add_run("Instructor: "), 11.5)
    r = p.add_run(data["instructor"])
    set_font(r, 11.5)
    r.italic = True
    paragraph(data["city"], 12, before=64, after=5)
    paragraph(data["year"], 12)
    return prefix


def configure_sections(doc, data, assets):
    cover, body = doc.sections[0], doc.sections[1]
    for section in [cover, body]:
        section.page_width, section.page_height = Inches(8.5), Inches(11)
        section.left_margin = section.right_margin = Inches(0.65)
        section.bottom_margin = Inches(0.65)
        section.header_distance = Inches(0)
        section.footer_distance = Inches(0.24)
        section.different_first_page_header_footer = False
        section.header.is_linked_to_previous = False
        section.footer.is_linked_to_previous = False
        clear_story(section.header)
        clear_story(section.footer)
        for story in [section.header, section.footer]:
            paragraph = story.paragraphs[0]
            paragraph.paragraph_format.space_before = Pt(0)
            paragraph.paragraph_format.space_after = Pt(0)
            paragraph.paragraph_format.line_spacing_rule = WD_LINE_SPACING.EXACTLY
            paragraph.paragraph_format.line_spacing = Pt(1)
    cover.top_margin = Inches(0)
    body.top_margin = Inches(0.97)
    body.start_type = WD_SECTION_START.NEW_PAGE
    p = body.header.paragraphs[0]
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.EXACTLY
    p.paragraph_format.line_spacing = Pt(1)
    scale = 8.5/2400
    anchored_picture(p, assets/"header_source.png", 0, 0, 468*scale, 160*scale,
                     roi=(0, 0, 468, 160), source_size=(2400, 160), picture_id=50200)
    for index, width in enumerate([802, 802, 328]):
        anchored_picture(p, assets/"header_source.png", (468+index*802)*scale, 0,
                         width*scale+0.01, 160*scale, roi=(480, 0, 480+min(width, 790), 160),
                         source_size=(2400, 160), picture_id=50201+index)
    text_box(p, [data["header_title"]], 468*scale, 0, (2400-468)*scale, 160*scale,
             15.5, "PolytechRunningTitle")
    footer = body.footer.paragraphs[0]
    footer.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
    footer.paragraph_format.space_before = Pt(4)
    footer.paragraph_format.space_after = Pt(0)
    footer.paragraph_format.tab_stops.add_tab_stop(Inches(7.1), WD_TAB_ALIGNMENT.RIGHT)
    borders = OxmlElement("w:pBdr")
    top = OxmlElement("w:top")
    for key, value in [("val", "single"), ("sz", "6"), ("space", "6"), ("color", "17579B")]:
        top.set(qn("w:"+key), value)
    borders.append(top)
    footer._p.get_or_add_pPr().append(borders)
    set_font(footer.add_run(data["authors"][0]), 9)
    set_font(footer.add_run("\tPage "), 9)
    field = OxmlElement("w:fldSimple")
    field.set(qn("w:instr"), "PAGE")
    footer._p.append(field)


def restyle_report(source, output=None, assets=ASSETS):
    source = Path(source)
    output = source if output is None else Path(output)
    assets = Path(assets)
    data = json.loads((assets/"metadata.json").read_text(encoding="utf-8"))
    doc = Document(source)
    if MARKER in (doc.core_properties.keywords or ""):
        if source != output:
            doc.save(output)
        return output
    body = doc._element.body
    elements = list(body)
    first = next((i for i, node in enumerate(elements)
                  if node.tag == qn("w:p") and "".join(node.itertext()).startswith("1 Problem Formulation")), None)
    if first is None:
        for i, node in enumerate(elements):
            if node.tag == qn("w:p") and "".join(node.xpath(".//w:t/text()")) == "1 Problem Formulation":
                first = i
                break
    if first is None:
        raise ValueError("Expected the Gradient Descent report produced by build_report.py.")
    abstract = []
    capture = False
    for p in doc.paragraphs:
        if p.text == "Objective":
            capture = True
            continue
        if capture:
            if p.text.startswith("Saint Petersburg") or p.text == "2026":
                break
            if p.text.strip():
                abstract.append(p.text)
    if not abstract:
        raise ValueError("Could not locate the report objective paragraphs.")
    original_tables = [[c.text for row in table.rows for c in row.cells] for table in doc.tables]
    original_body_text = ["".join(node.xpath(".//w:t/text()")) for node in elements[first:]
                          if node.tag == qn("w:p")]
    for node in elements[:first]:
        body.remove(node)
    apply_styles(doc)
    prefix = add_cover(doc, data, assets)
    doc.add_section(WD_SECTION_START.NEW_PAGE)
    section_paragraph = doc.paragraphs[-1]._p
    prefix.append(section_paragraph)
    h = doc.add_heading("Abstract", 1)
    prefix.append(h._p)
    for text in abstract:
        prefix.append(doc.add_paragraph(text)._p)
    p = doc.add_paragraph()
    p.add_run().add_break(WD_BREAK.PAGE)
    prefix.append(p._p)
    for node in prefix:
        node.getparent().remove(node)
    for i, node in enumerate(prefix):
        body.insert(i, node)
    configure_sections(doc, data, assets)
    doc.core_properties.title = "Gradient Descent for Two Unconstrained Objectives"
    doc.core_properties.author = "; ".join(data["authors"])
    doc.core_properties.keywords = MARKER
    final_tables = [[c.text for row in table.rows for c in row.cells] for table in doc.tables]
    if final_tables != original_tables:
        raise RuntimeError("Table contents changed during formatting.")
    paragraphs = list(doc.paragraphs)
    first_body = next(i for i, p in enumerate(paragraphs) if p.text == "1 Problem Formulation")
    final_text = [p.text for p in paragraphs[first_body:]]
    if final_text != original_body_text:
        raise RuntimeError("Body text changed during formatting.")
    output.parent.mkdir(parents=True, exist_ok=True)
    doc.save(output)
    return output


def main():
    parser = argparse.ArgumentParser(description="Apply the supplied Polytech report theme")
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    path = restyle_report(args.source, args.output)
    print("Polytech report written to "+str(path))


if __name__ == "__main__":
    main()
