"""CRUD operations for chunk entities using FastCRUD."""

from fastcrud import FastCRUD

from .models import Chunk

chunk_crud: FastCRUD = FastCRUD(Chunk)
