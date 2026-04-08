import io, time
from PIL import Image
from fpdf import FPDF

SEVERITY_RGB = {
    "HIGH":   (239, 68,  68),
    "MEDIUM": (249, 115, 22),
    "LOW":    (234, 179,  8),
    "NONE":   (34,  197, 94),
}

class MedVisionPDF(FPDF):
    def __init__(self, report, user_name, user_role):
        super().__init__()
        self.report    = report
        self.user_name = user_name
        self.user_role = user_role
        self.set_margins(20, 24, 20)
        self.set_auto_page_break(auto=True, margin=20)

    def header(self):
        self.set_fill_color(6, 12, 24)
        self.rect(0, 0, 210, 22, 'F')
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(0, 220, 255)
        self.set_y(6)
        self.cell(0, 8, "MedVision AI  —  Clinical Scan Report", align="C")
        self.set_font("Helvetica", "", 7)
        self.set_text_color(80, 100, 130)
        self.set_y(14)
        self.cell(0, 5, f"CONFIDENTIAL  |  {self.report['timestamp']}  |  {self.user_name} ({self.user_role})", align="C")
        self.ln(10)

    def footer(self):
        self.set_y(-13)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(80, 100, 130)
        self.cell(0, 5, "MedVision AI is an assistive tool — not a substitute for professional medical judgement.", align="C")
        self.ln(3)
        self.cell(0, 4, f"Page {self.page_no()}", align="C")

    def section_title(self, title):
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(0, 180, 210)
        self.set_fill_color(8, 15, 30)
        self.cell(0, 7, f"  {title}", fill=True, ln=True)
        self.set_draw_color(0, 100, 140)
        self.set_line_width(0.3)
        self.line(20, self.get_y(), 190, self.get_y())
        self.ln(3)

    def kv(self, key, value, bold=False):
        self.set_font("Helvetica", "", 9)
        self.set_text_color(90, 110, 140)
        self.cell(50, 7, key)
        self.set_font("Helvetica", "B" if bold else "", 9)
        self.set_text_color(210, 225, 240)
        self.cell(0, 7, str(value), ln=True)

    def status_banner(self, status, severity):
        color = SEVERITY_RGB.get(severity, (100,100,100))
        self.set_fill_color(*color)
        self.set_text_color(255,255,255)
        self.set_font("Helvetica", "B", 11)
        icon = "!" if status == "ABNORMAL" else "OK"
        self.cell(0, 11, f"  [{icon}]  STATUS: {status}   SEVERITY: {severity}", fill=True, ln=True)
        self.ln(3)

    def finding_block(self, f, idx):
        color = SEVERITY_RGB.get(f["severity"], (100,100,100))
        bg = (10,18,32) if idx%2==0 else (8,14,26)
        self.set_fill_color(*bg)
        self.rect(20, self.get_y(), 170, 20, 'F')
        self.set_fill_color(*color)
        self.rect(20, self.get_y(), 3, 20, 'F')

        y = self.get_y() + 3
        self.set_xy(25, y)
        self.set_font("Helvetica","B",9)
        self.set_text_color(215,230,245)
        self.cell(38, 5, f["label"])

        bx, by, bw = 65, y+1, 55
        self.set_fill_color(18,32,55)
        self.rect(bx, by, bw, 3, 'F')
        self.set_fill_color(*color)
        self.rect(bx, by, bw * f["confidence"], 3, 'F')

        self.set_xy(bx+bw+2, y)
        self.set_font("Helvetica","",8)
        self.set_text_color(*color)
        self.cell(16, 5, f"{f['confidence']:.0%}")

        self.set_xy(148, y)
        self.set_font("Helvetica","B",8)
        self.cell(38, 5, f["severity"], align="R")

        self.set_xy(25, y+7)
        self.set_font("Helvetica","I",7.5)
        self.set_text_color(90,110,140)
        desc = f["description"]
        self.cell(162, 4, desc[:95]+("..." if len(desc)>95 else ""))
        self.ln(22)


def generate_pdf(report, annotated_image, original_image, user_name, user_role):
    pdf = MedVisionPDF(report, user_name, user_role)
    pdf.add_page()
    # Dark background
    pdf.set_fill_color(6,12,24)
    pdf.rect(0,22,210,275,'F')

    pdf.section_title("PATIENT INFORMATION")
    pdf.kv("Patient ID:", report["patient_id"], bold=True)
    pdf.kv("Scan Type:", report["scan_type"])
    pdf.kv("Timestamp:", report["timestamp"])
    pdf.kv("Reviewed by:", f"{user_name} ({user_role})")
    pdf.ln(3)

    pdf.section_title("OVERALL ASSESSMENT")
    pdf.status_banner(report["overall_status"], report["overall_severity"])
    high   = sum(1 for f in report["findings"] if f["severity"]=="HIGH")
    medium = sum(1 for f in report["findings"] if f["severity"]=="MEDIUM")
    low    = sum(1 for f in report["findings"] if f["severity"]=="LOW")
    avg_c  = sum(f["confidence"] for f in report["findings"])/len(report["findings"]) if report["findings"] else 0
    pdf.kv("Total Findings:", str(report["total_findings"]))
    pdf.kv("High / Medium / Low:", f"{high} / {medium} / {low}")
    pdf.kv("Avg Confidence:", f"{avg_c:.1%}")
    pdf.ln(3)

    if report["findings"]:
        pdf.section_title("DETAILED FINDINGS")
        for i, f in enumerate(report["findings"]):
            pdf.finding_block(f, i)
        pdf.ln(2)

    pdf.section_title("CLINICAL RECOMMENDATIONS")
    pdf.set_font("Helvetica","",8.5)
    pdf.set_text_color(170,190,215)
    if report["overall_status"] == "NORMAL":
        pdf.multi_cell(0, 6, "No significant abnormalities detected. Routine follow-up as per standard protocol.")
    else:
        recs = []
        for f in report["findings"]:
            if f["severity"]=="HIGH":   recs.append(f"  * {f['label']}: Urgent specialist referral. Do not delay.")
            elif f["severity"]=="MEDIUM": recs.append(f"  * {f['label']}: Clinical correlation within 1-2 weeks.")
            else:                         recs.append(f"  * {f['label']}: Follow-up imaging in 3-6 months.")
        pdf.multi_cell(0, 7, "\n".join(recs))
    pdf.ln(3)

    # Page 2: Images
    pdf.add_page()
    pdf.set_fill_color(6,12,24)
    pdf.rect(0,22,210,275,'F')
    pdf.section_title("SCAN IMAGES")

    orig_path = "/tmp/mv_orig.jpg"
    ann_path  = "/tmp/mv_ann.jpg"
    original_image.save(orig_path, "JPEG", quality=85)
    annotated_image.save(ann_path,  "JPEG", quality=85)

    y = pdf.get_y()
    pdf.set_font("Helvetica","B",8)
    pdf.set_text_color(80,100,130)
    pdf.cell(85, 6, "ORIGINAL SCAN", align="C")
    pdf.cell(10, 6, "")
    pdf.cell(85, 6, "AI ANNOTATED SCAN", align="C", ln=True)
    pdf.image(orig_path, x=20,  y=pdf.get_y(), w=80)
    pdf.image(ann_path,  x=110, y=pdf.get_y(), w=80)
    pdf.ln(86)

    pdf.set_font("Helvetica","I",7.5)
    pdf.set_text_color(80,100,130)
    pdf.multi_cell(0, 5, "Left: Original scan. Right: AI annotations (Red=HIGH, Orange=MEDIUM, Yellow=LOW severity).")

    return pdf.output()
