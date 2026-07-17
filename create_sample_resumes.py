from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from pathlib import Path


def create_sample_resume_1(output_path):
    c = canvas.Canvas(str(output_path), pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 18)
    c.drawString(100, height - 100, "John Doe")

    c.setFont("Helvetica", 12)
    c.drawString(100, height - 130, "Email: john.doe@email.com | Phone: (555) 123-4567")

    c.setFont("Helvetica-Bold", 14)
    c.drawString(100, height - 180, "Summary")
    c.setFont("Helvetica", 12)
    c.drawString(100, height - 205, "Experienced data professional with 5+ years in Python, SQL, and Machine Learning.")

    c.setFont("Helvetica-Bold", 14)
    c.drawString(100, height - 250, "Skills")
    c.setFont("Helvetica", 12)
    c.drawString(100, height - 275, "Python, SQL, Machine Learning, AWS, Data Analysis, Pandas, NumPy")

    c.save()


def create_sample_resume_2(output_path):
    c = canvas.Canvas(str(output_path), pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 18)
    c.drawString(100, height - 100, "Jane Smith")

    c.setFont("Helvetica", 12)
    c.drawString(100, height - 130, "Email: jane.smith@email.com | Phone: (555) 987-6543")

    c.setFont("Helvetica-Bold", 14)
    c.drawString(100, height - 180, "Summary")
    c.setFont("Helvetica", 12)
    c.drawString(100, height - 205, "Marketing specialist with experience in social media and content creation.")

    c.setFont("Helvetica-Bold", 14)
    c.drawString(100, height - 250, "Skills")
    c.setFont("Helvetica", 12)
    c.drawString(100, height - 275, "Social Media, Content Writing, SEO, Google Analytics")

    c.save()


if __name__ == "__main__":
    test_folder = Path("test_resumes")
    test_folder.mkdir(exist_ok=True)

    create_sample_resume_1(test_folder / "John_Doe_Resume.pdf")
    create_sample_resume_2(test_folder / "Jane_Smith_Resume.pdf")
    print("Sample resumes created successfully!")
