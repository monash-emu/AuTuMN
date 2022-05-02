import os
import json

from fpdf import FPDF
from autumn.settings import DOCS_PATH
from autumn.settings import PROJECTS_PATH
from datetime import datetime

from autumn.tools.utils.s3 import upload_file_s3, get_s3_client

POLICY_PATH = os.path.join(DOCS_PATH, "papers", "covid_19", "policy_brief")
POLICY_PDF = os.path.join(POLICY_PATH, "policy_brief.pdf")
POLICY_PHL = os.path.join(
    PROJECTS_PATH, "covid_19", "philippines", "philippines", "policy_brief.json"
)
POLICY_MYS = os.path.join(PROJECTS_PATH, "covid_19", "malaysia", "malaysia", "policy_brief.json")
POLICY_LKA = os.path.join(PROJECTS_PATH, "covid_19", "sri_lanka", "sri_lanka", "policy_brief.json")

POLICY_JSON = POLICY_LKA

AUTUMN_LOGO = os.path.join(POLICY_PATH, "images", "AuTuMN_official.png")
MONASH_LOGO = os.path.join(POLICY_PATH, "images", "Monash_University_logo.png")
PAGE_BG = os.path.join(POLICY_PATH, "images", "page_background.png")
WHO_LOGO = os.path.join(POLICY_PATH, "images", "WHO_SEARO.jpg")

flag_path = {
    "australia": os.path.join(POLICY_PATH, "images", "AUS.png"),
    "indonesia": os.path.join(POLICY_PATH, "images", "IDN.png"),
    "sri_lanka": os.path.join(POLICY_PATH, "images", "LKA.png"),
    "malaysia": os.path.join(POLICY_PATH, "images", "MYS.png"),
    "nepal": os.path.join(POLICY_PATH, "images", "NPL.png"),
    "philippines": os.path.join(POLICY_PATH, "images", "PHL.png"),
    "vietnam": os.path.join(POLICY_PATH, "images", "VNM.png"),
}

dark_blue = (0, 0, 55)
blue = (115, 152, 181)
light_blue = (219, 228, 233)
dark_red = (130, 0, 0)
orange = (217, 71, 0)
yellow = (244, 172, 69)
white = (255, 255, 255)
black = (0, 0, 0)


class PDF(FPDF):
    def header(self):

        # self.c_margin = self.c_margin +4

        # helvetica bold 15
        self.set_font("helvetica", "B", 12)
        self.set_margins(0, 0, 0)

        self.image(PAGE_BG, 0, 0)

        # Title
        self.set_fill_color(*dark_blue)
        self.set_draw_color(*dark_blue)
        self.set_text_color(*white)

        self.cell(
            w=0,
            h=8,
            txt="Modelling projections of the Covid-19 epidemic in",
            ln=1,
            border=1,
            align="C",
            fill=True,
        )

        self.cell(
            w=0,
            h=8,
            txt=self.region.replace("_", " ").title() + " (" + self.date + ")",
            ln=1,
            border=0,
            align="C",
            fill=True,
        )

        # Logo
        self.image(MONASH_LOGO, 2, 2, h=12)
        self.image(AUTUMN_LOGO, 161, 2, h=11)

        # Set margins
        self.set_margins(5, 20, 5)

    # Page footer
    def footer(self):
        # Colour
        self.set_fill_color(*dark_blue)
        self.set_draw_color(*dark_blue)
        self.set_text_color(*white)

        # Position at 1.5 cm from bottom
        self.set_margins(0, 0)
        self.set_y(-16)
        # helvetica italic 8
        self.set_font("helvetica", "I", 8)

        # Page number
        self.cell(0, 16, "Page " + str(self.page_no()) + "/{nb}", 0, 0, "C", fill=True)

        self.image(flag_path[self.region], 2, 283, h=12)
        self.image(WHO_LOGO, 178, 283, h=12)

    def read_policy_brief(self, policy_json):

        with open(policy_json) as file:
            self.pb = json.load(file)

    def extract_attributes(self):

        self.region = self.pb["RUN_ID"].split("/")[1]
        self.date = str(datetime.fromtimestamp(int(self.pb["RUN_ID"].split("/")[2])).date())
        self.heading = [*self.pb][1:]
        self.output_pdf = os.path.join(POLICY_PATH, f"policy_brief_{self.region}.pdf")

    def print_text(self, width, heading):

        if heading.lower() == "abbreviations":
            x, y = self.get_x(), self.get_y()
            self.set_xy(x, 20)

        self.print_title(heading)

        if heading.lower() == "abbreviations":
            text = "".join(f"{each}\n" for each in self.pb[heading])
            self.set_fill_color(*blue)
            self.set_text_color(*white)
            self.multi_cell(width, 5, txt=text, align="L", ln=1, fill=True)

        else:
            self.multi_cell(width, 5, self.pb[heading], align="L", ln=3)

    def print_title(self, heading):

        # Title
        self.set_font("helvetica", "B", 10)
        self.set_text_color(*orange)
        self.cell(txt=heading, ln=2)

        # Set body text location & colour
        x, y = self.get_x(), self.get_y()
        self.set_xy(x, y + 2)

        self.set_font("helvetica", "", 10)
        self.set_text_color(*black)

    def print_dict(self, width, heading, background=None):

        text_list = []
        txt_key = [key for key in [*self.pb[heading]] if "_pic" not in key]

        img_list = []
        img_list = [key for key in [*self.pb[heading]] if "_pic" in key]

        for txt in txt_key:
            text_list.append(self.pb[heading][txt])

        text = "".join(f"{txt}\n" for txt in text_list)

        self.print_title(heading)

        body_start_y = self.get_y()

        if background is not None:
            self.set_fill_color(*background)
            self.multi_cell(width, h=5, txt=text, align="L", fill=True, ln=1)
        else:
            self.multi_cell(width, h=5, txt=text, align="L", ln=1)

        body_end_y = self.get_y()
        body_end_x = self.get_x()

        self.set_xy(width + 5, body_start_y)
        for img in img_list:
            self.image(self.pb[heading][img], h=50)

        img_end_y = self.get_y()

        body_end_y = body_end_y if body_end_y > img_end_y else img_end_y

        if len(img_list):
            self.set_xy(body_end_x, body_end_y)
        else:
            self.set_xy(body_end_x, body_end_y + 5)

    def print_list(self, width, heading, background=None):

        if heading.lower() == "abbreviations":
            x, y = self.get_x(), self.get_y()
            self.set_xy(x, 20)

        self.print_title(heading)

        text = "".join(f"{each}\n" for each in self.pb[heading])

        if background is not None:
            self.set_fill_color(*background)
            fill = True
        else:
            fill = False

        self.set_text_color(*white)
        self.multi_cell(width, 5, txt=text, align="L", ln=1, fill=fill)

    def upload_pdf(self):

        dest_key = self.pb["RUN_ID"]
        dest_key = f"{dest_key}/plots/policy_brief.pdf"

        s3_client = get_s3_client()
        upload_file_s3(s3_client, self.output_pdf, dest_key)


pdf = PDF(orientation="P", unit="mm", format="A4")

# This also works
# pdf.set_auto_page_break(True,30)

pdf.read_policy_brief(POLICY_JSON)
pdf.extract_attributes()
pdf.add_page()

pdf.print_text(140, pdf.heading[0])
pdf.print_list(60, pdf.heading[1], background=(blue))
pdf.print_dict(0, pdf.heading[2], background=(yellow))
pdf.print_dict(140, pdf.heading[3])
pdf.ln(60)  # Not ideal - add line break to manage automatic page breaks
pdf.print_dict(140, pdf.heading[4])
pdf.print_text(0, pdf.heading[5])
pdf.output(pdf.output_pdf)
pdf.upload_pdf()
