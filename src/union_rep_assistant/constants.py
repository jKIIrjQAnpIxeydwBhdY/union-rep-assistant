from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

# TODO: these paths are britle and will break if things are moved around
SECURITY_CONTRACT_PATH = Path(
    "contract/2024-2028-RAB-Security-Officers-Owners-Agreement-Executed.pdf"
)

TXT_PATH = Path("2024-2028-RAB-Security-Officers-Owners-Agreement-Executed.txt")

TEXT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # chunk size is 1000
    chunk_overlap=200,  # each chunk has an overlap of characters of 200
)
