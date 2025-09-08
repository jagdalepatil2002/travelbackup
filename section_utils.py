import re
from typing import List, Dict

def split_markdown_sections(markdown_text: str) -> List[Dict[str, str]]:
    """
    Splits markdown text into a list of sections with 'title' and 'content'.
    Assumes each section starts with a '## ' heading.
    """
    pattern = r'(^## .*$)'  # Matches headings
    parts = re.split(pattern, markdown_text, flags=re.MULTILINE)
    sections = []
    i = 1
    while i < len(parts):
        title = parts[i].strip()
        content = parts[i+1].strip() if (i+1) < len(parts) else ''
        sections.append({'title': title, 'content': content})
        i += 2
    return sections

def paginate_sections(sections: List[Dict[str, str]], max_sections_per_page: int = 3, max_chars_per_page: int = 6000) -> List[List[Dict[str, str]]]:
    """
    Groups sections into pages, with up to max_sections_per_page per page.
    If a page would exceed max_chars_per_page, reduce the number of sections on that page.
    """
    pages = []
    i = 0
    while i < len(sections):
        page = []
        chars = 0
        for j in range(max_sections_per_page):
            if i + j >= len(sections):
                break
            section = sections[i + j]
            section_len = len(section['title']) + len(section['content'])
            if chars + section_len > max_chars_per_page and j > 0:
                break
            page.append(section)
            chars += section_len
        pages.append(page)
        i += len(page)
    return pages
