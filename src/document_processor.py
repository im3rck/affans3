"""Document Processing Module for PDFs and CSV files"""
import os
from typing import List, Dict
import pandas as pd
import pymupdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document


class DocumentProcessor:
    """Process and chunk documents from various sources"""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def process_pdf(self, file_path: str) -> List[Document]:
        """Extract text from PDF and create document chunks"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        documents = []
        doc = pymupdf.open(file_path)

        for page_num, page in enumerate(doc):
            text = page.get_text()
            if text.strip():
                metadata = {
                    "source": file_path,
                    "page": page_num + 1,
                    "document_type": "pdf",
                    "filename": os.path.basename(file_path)
                }
                documents.append(Document(page_content=text, metadata=metadata))

        doc.close()

        # Split into chunks
        chunks = self.text_splitter.split_documents(documents)
        return chunks

    def process_csv(self, file_path: str) -> List[Document]:
        """Process CSV file and create document chunks"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CSV file not found: {file_path}")

        df = pd.read_csv(file_path)
        documents = []

        for idx, row in df.iterrows():
            # Create a structured text representation
            text_parts = []

            # Add product information
            if 'product_name' in df.columns:
                text_parts.append(f"Product: {row['product_name']}")

            if 'category' in df.columns:
                text_parts.append(f"Category: {row['category']}")

            if 'about_product' in df.columns and pd.notna(row['about_product']):
                text_parts.append(f"Description: {row['about_product']}")

            if 'rating' in df.columns:
                text_parts.append(f"Rating: {row['rating']}/5")

            if 'rating_count' in df.columns:
                text_parts.append(f"Number of Ratings: {row['rating_count']}")

            # Add price information
            if 'discounted_price' in df.columns:
                text_parts.append(f"Price: {row['discounted_price']}")

            if 'discount_percentage' in df.columns:
                text_parts.append(f"Discount: {row['discount_percentage']}")

            # Add review information
            if 'review_title' in df.columns and pd.notna(row['review_title']):
                text_parts.append(f"Review Title: {row['review_title']}")

            if 'review_content' in df.columns and pd.notna(row['review_content']):
                text_parts.append(f"Review: {row['review_content']}")

            if 'user_name' in df.columns and pd.notna(row['user_name']):
                text_parts.append(f"Reviewer: {row['user_name']}")

            text = "\n".join(text_parts)

            if text.strip():
                metadata = {
                    "source": file_path,
                    "row_index": idx,
                    "document_type": "csv",
                    "product_id": row.get('product_id', 'unknown')
                }

                # Add additional metadata
                if 'product_name' in df.columns:
                    metadata['product_name'] = str(row['product_name'])
                if 'category' in df.columns:
                    metadata['category'] = str(row['category'])

                documents.append(Document(page_content=text, metadata=metadata))

        # Split into chunks if needed
        chunks = self.text_splitter.split_documents(documents)
        return chunks

    def process_all_documents(self, pdf_files: List[str], csv_files: List[str]) -> List[Document]:
        """Process all documents and return combined chunks"""
        all_chunks = []

        print("Processing PDF documents...")
        for pdf_file in pdf_files:
            try:
                chunks = self.process_pdf(pdf_file)
                all_chunks.extend(chunks)
                print(f"✓ Processed {pdf_file}: {len(chunks)} chunks")
            except Exception as e:
                print(f"✗ Error processing {pdf_file}: {str(e)}")

        print("\nProcessing CSV documents...")
        for csv_file in csv_files:
            try:
                chunks = self.process_csv(csv_file)
                all_chunks.extend(chunks)
                print(f"✓ Processed {csv_file}: {len(chunks)} chunks")
            except Exception as e:
                print(f"✗ Error processing {csv_file}: {str(e)}")

        print(f"\nTotal chunks created: {len(all_chunks)}")
        return all_chunks


if __name__ == "__main__":
    # Test the processor
    processor = DocumentProcessor()

    pdf_files = [
        "Amazon_Customer_Service_Operations_KB.pdf",
        "Amazon_HR_Employee_Support_KB.pdf",
        "Amazon_IT_Device_Support_KB.pdf"
    ]

    csv_files = ["amazon.csv"]

    chunks = processor.process_all_documents(pdf_files, csv_files)
    print(f"\nSample chunk:\n{chunks[0].page_content[:200]}...")
    print(f"Metadata: {chunks[0].metadata}")
