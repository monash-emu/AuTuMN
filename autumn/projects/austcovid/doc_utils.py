import pylatex as pl
from pylatex import LineBreak
from pylatex.section import Section
from pylatex.utils import NoEscape, bold


class DocElement:
    """
    Abstract class for creating a model with accompanying TeX documentation.
    """
    def __init__():
        pass

    def emit_latex():
        pass


class TextElement(DocElement):
    """
    Write text input to TeX document using PyLaTeX commands.
    """
    def __init__(
            self, 
            text: str,
        ):
        """
        Set up object with text input.

        Args:
            text: The text to write
        """
        self.text = NoEscape(text) if "\cite{" in text else text

    def emit_latex(
            self, 
            doc: pl.document.Document,
        ):
        """
        Write the text to the document.

        Args:
            doc: The PyLaTeX object to add to
        """
        doc.append(self.text)


class FigElement(DocElement):
    """
    Add a figure to a TeX document using PyLaTeX commands.
    """
    def __init__(
            self, 
            fig_name: str,
            caption: str="",
            resolution: str="350px",
        ):
        """
        Set up object with figure input and other requests.

        Args:
            fig_name: The name of the figure to write
            caption: Figure caption
            resolution: Resolution to write to
        """
        self.fig_name = fig_name
        self.caption = caption
        self.resolution = resolution
    
    def emit_latex(
            self, 
            doc: pl.document.Document,
        ):
        """
        Write the figure to the document.

        Args:
            doc: The PyLaTeX object to add to
        """
        with doc.create(pl.Figure()) as plot:
            plot.add_image(self.fig_name, width=self.resolution)
            plot.add_caption(self.caption)


class TableElement(DocElement):
    
    def __init__(self, col_widths, headers, rows):
        self.col_widths = col_widths
        self.headers = headers
        self.rows = rows

    def emit_latex(self, doc):
        with doc.create(pl.Tabular(self.col_widths)) as calibration_table:
            calibration_table.add_hline()
            calibration_table.add_row([bold(i) for i in self.headers])
            for row in self.rows:
                calibration_table.add_hline()
                calibration_table.add_row(row)
            calibration_table.add_hline()
        doc.append(LineBreak())


class DocumentedProcess:

    def __init__(self, doc, add_documentation):
        self.doc = doc
        self.add_documentation = add_documentation
        self.doc_sections = {}

    def add_element_to_doc(
        self, 
        section_name: str, 
        element: DocElement,
    ):
        """
        Add a new element to the list of elements to be included in a document section.

        Args:
            section_name: Name of the section to add to
            element: The object to include in the section
        """
        if section_name not in self.doc_sections:
            self.doc_sections[section_name] = []
        self.doc_sections[section_name].append(element)

    def compile_doc(self):
        """
        Apply all the document elements to the document,
        looping through each section and using each element's emit_latex method.
        """
        for section in self.doc_sections:
            with self.doc.create(Section(section)):
                for element in self.doc_sections[section]:
                    element.emit_latex(self.doc)
