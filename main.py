import asyncio
import traceback

import flet as ft

from src.ui.ui import UI
from src.logic.uiStateReader import state_reader, close

async def main(page: ft.Page):
    # page.theme_mode = "dark"
    # page.theme = ft.Theme(color_scheme_seed=ft.Colors.GREY)
    ui = UI()
    page.add(ui)

    # при закрытии окна освобождаем камеру, иначе она останется занятой
    page.on_close = lambda e: close()

    while True:
        try:
            await state_reader(ui)
        except Exception:
            # раньше любое исключение закрывало окно/вешало цикл
            traceback.print_exc()
            await asyncio.sleep(1)
        # ui.cv_window.update()
        await asyncio.sleep(0.05)



if __name__ == "__main__":
    ft.run(main)
