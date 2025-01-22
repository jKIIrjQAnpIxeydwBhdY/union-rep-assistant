import io
import base64
from pathlib import Path

import fitz
from PIL import Image
from langchain_core.messages import HumanMessage, SystemMessage


def pdf_page_to_base64(pdf_path: Path):
    pdf_document = fitz.open(pdf_path)
    pdf_images = []
    for page_index in range(len(pdf_document)):
        page = pdf_document.load_page(page_index - 1)  # input is one-indexed
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        pdf_images.append(base64.b64encode(buffer.getvalue()).decode("utf-8"))
    return pdf_images


def get_pdf_text(model, b64_img):
    """this ocr function makes mistakes , translated: 32BJ to 3281 but I think useful for a draft"""

    messages = [
        SystemMessage(
            content="You are an expert in image OCR. Given an image of text, respond with the exact text in the image only. Do not add or remove words."
        ),
        HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": "Extract the text from this image. Maintain spelling exactly. Output no other content.",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"},
                },
            ]
        ),
    ]
    response = model.invoke([messages])
    return response.content


def convert_contract_to_string_list(contract_path: Path) -> list[str]:
    with contract_path.open("r", encoding="utf-8") as file:
        content = file.read()
        return content.split(
            "||"
        )  # I intentionally added this delimiter to the contract


def save_ocr_output(contract_path: Path, ocr: list[str], delimiter="||") -> None:
    contract_path.write_text(delimiter.join(ocr))


def generate_contract_text(pdf_path: Path, txt_path: Path, llm) -> None:
    b64images = pdf_page_to_base64(pdf_path)
    docs = [get_pdf_text(llm, img) for img in b64images]
    # takes around 10 minutes

    save_ocr_output(txt_path, docs, "||")
