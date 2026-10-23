import asyncio
import flet as ft

from src.ui.ui import UI
from src.logic.uiStateReader import state_reader

async def main(page: ft.Page):
    # page.theme_mode = "dark"
    # page.theme = ft.Theme(color_scheme_seed=ft.Colors.GREY)
    ui = UI()
    page.add(ui)

    while True:
        await state_reader(ui)
        # ui.cv_window.update()
        await asyncio.sleep(0.05) 
    


if __name__ == "__main__":
    ft.run(main)
    