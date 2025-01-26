import io
import base64
from pathlib import Path
from typing import Any

import pymupdf
from PIL import Image
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from .constants import TEXT_SPLITTER


def pdf_page_to_base64(pdf_path: Path) -> list[str]:
    """
    Convert each page of a PDF document to a base64-encoded PNG image.

    This function processes each page of a given PDF file, converts it to an image,
    and then encodes the image in base64 format. The resulting base64-encoded images
    are returned as a list of strings.

    Args:
        pdf_path (Path): The path to the PDF file to be converted.

    Returns:
        List[str]: A list of base64-encoded strings, where each string represents
                   a page of the PDF as a PNG image.

    Raises:
        FileNotFoundError: If the specified PDF file does not exist.
        ValueError: If the PDF cannot be opened or processed.
    """
    pdf_document = pymupdf.open(pdf_path)
    pdf_images = []
    for page_index in range(len(pdf_document)):
        page = pdf_document.load_page(page_index - 1)  # input is one-indexed
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        pdf_images.append(base64.b64encode(buffer.getvalue()).decode("utf-8"))
    return pdf_images


def get_pdf_text(model: Any, b64_img: str) -> str:
    """
    Perform OCR (Optical Character Recognition) on a base64-encoded image of text using a provided model.

    This function uses an OCR model to extract text from a base64-encoded image.
    While the OCR model may make mistakes (e.g., translating "32BJ" to "3281"),
    it is sufficient for a Minimum Viable Product (MVP).

    Args:
        model (Any): The OCR model to invoke for text extraction. It should support the `invoke` method.
        b64_img (str): A base64-encoded string representing the image to extract text from.

    Returns:
        str: The exact text extracted from the image as interpreted by the OCR model.

    Raises:
        ValueError: If the OCR model fails to process the image or returns an invalid response.
    """
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
    response = model.invoke(messages)
    return response.content


def convert_contract_to_string_list(contract_path: Path) -> list[str]:
    with contract_path.open("r", encoding="utf-8") as file:
        content = file.read()
        return content.split(
            "||"
        )  # I intentionally added this delimiter to the contract


def save_ocr_output(save_path: Path, ocr: list[str], delimiter="||") -> None:
    save_path.write_text(delimiter.join(ocr))


def generate_contract_text(pdf_path: Path, save_path: Path, llm) -> list[str]:
    b64images = pdf_page_to_base64(pdf_path)
    docs = [get_pdf_text(llm, img) for img in b64images]
    # takes around 10 minutes
    save_ocr_output(save_path, docs, "||")
    return docs


def create_vector_store(docs_with_page_number, faiss_index_directory: Path):
    texts = TEXT_SPLITTER.split_documents(docs_with_page_number)
    vector_store = FAISS.from_documents(texts, OpenAIEmbeddings())
    # to create vectore store for the first time.  has a decent cost
    # TODO: does this require an api key?
    vector_store.save_local(faiss_index_directory)  # save it to avoid future cost
    return vector_store
