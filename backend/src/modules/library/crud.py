"""CRUD operations for library entities using FastCRUD."""

from fastcrud import FastCRUD

from .models import Library

library_crud: FastCRUD = FastCRUD(Library)
