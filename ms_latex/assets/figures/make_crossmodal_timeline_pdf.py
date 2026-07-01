from pathlib import Path

from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


HERE = Path(__file__).resolve().parent
PNG = HERE / "crossmodalTimeline.png"
PDF = HERE / "crossmodalTimeline.pdf"


def main():
    image = ImageReader(str(PNG))
    width, height = image.getSize()
    pdf = canvas.Canvas(str(PDF), pagesize=(width, height))
    pdf.drawImage(image, 0, 0, width=width, height=height)
    pdf.showPage()
    pdf.save()


if __name__ == "__main__":
    main()
