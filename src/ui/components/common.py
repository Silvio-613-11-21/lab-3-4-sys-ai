"""Мелкие переиспользуемые элементы интерфейса (карточки, заголовки)."""

from __future__ import annotations

import flet as ft


def section_title(text: str) -> ft.Text:
    """Заголовок группы настроек."""
    return ft.Text(
        text,
        size=13,
        weight=ft.FontWeight.BOLD,
        color=ft.Colors.LIGHT_BLUE_200,
    )


def card(*controls: ft.Control, width: int | None = None, spacing: int = 10) -> ft.Container:
    """Карточка с рамкой — визуальный «блок настроек»."""
    return ft.Container(
        content=ft.Column(list(controls), spacing=spacing, tight=True),
        width=width,
        padding=12,
        border=ft.Border.all(1, ft.Colors.GREY_700),
        border_radius=10,
        bgcolor=ft.Colors.GREY_900,
    )


def hint(text: str) -> ft.Text:
    """Мелкая серая подсказка."""
    return ft.Text(text, size=11, color=ft.Colors.GREY_400, italic=True)


def path_label(text: str = "файл не выбран") -> ft.Text:
    """Подпись выбранного файла (обрезается по ширине)."""
    return ft.Text(
        text,
        size=11,
        color=ft.Colors.GREY_300,
        max_lines=2,
        overflow=ft.TextOverflow.ELLIPSIS,
    )


def file_button(text: str, on_click, icon=ft.Icons.FOLDER_OPEN) -> ft.Button:
    """Кнопка выбора файла."""
    return ft.Button(content=text, icon=icon, on_click=on_click)
