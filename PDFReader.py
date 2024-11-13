import PyPDF2

class PDFReader:
    def __init__(self, file_path):
        self.file_path = file_path

    def read_pdf(self):
        text = ""
        with open(self.file_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text()
        return text
