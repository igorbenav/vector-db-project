"""CRUD operations for document entities using FastCRUD."""

from fastcrud import FastCRUD

from .models import Document

document_crud: FastCRUD = FastCRUD(Document)
