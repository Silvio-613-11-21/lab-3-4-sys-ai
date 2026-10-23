"""Точка входа приложения.

Запуск:

    python main.py

Приложение открывается как настольное окно Flet. Если что-то пошло не так
на старте (например, не установлен ``flet-desktop`` или несовместимые
версии библиотек), вместо «молчаливого» закрытия окна на экран выводится
понятное сообщение об ошибке — это сильно упрощает диагностику.
"""

from __future__ import annotations

import traceback

import flet as ft


def main(page: ft.Page) -> None:
    """Создать интерфейс приложения на переданной странице Flet."""
    try:
        from src.ui.app import build

        build(page)
    except Exception as error:  # noqa: BLE001 - показываем ошибку пользователю
        traceback.print_exc()
        page.bgcolor = ft.Colors.GREY_900
        page.add(
            ft.Column(
                controls=[
                    ft.Text("Не удалось запустить приложение", size=22,
                            color=ft.Colors.RED_300, weight=ft.FontWeight.BOLD),
                    ft.Text(str(error), size=14, color=ft.Colors.WHITE),
                    ft.Text(
                        "Проверьте, что установлены зависимости из requirements.txt "
                        "и запустите «python main.py» из корня проекта. "
                        "Подробности — в консоли и в README.",
                        size=12,
                        color=ft.Colors.GREY_400,
                    ),
                ],
                spacing=12,
            )
        )


def _parse_arguments(argv: list[str]) -> dict:
    """Разобрать аргументы командной строки без сторонних библиотек.

    ``python main.py``                  — настольное окно;
    ``python main.py --web``            — браузер, порт 8550;
    ``python main.py --web --port 9000`` — браузер на другом порту;
    ``python main.py --view flet_app``  — явно настольный режим.
    """
    options = {"web": False, "port": 8550, "host": "0.0.0.0"}
    index = 0
    while index < len(argv):
        argument = argv[index]
        if argument == "--web":
            options["web"] = True
        elif argument == "--port" and index + 1 < len(argv):
            index += 1
            options["port"] = int(argv[index])
        elif argument == "--host" and index + 1 < len(argv):
            index += 1
            options["host"] = argv[index]
        elif argument == "--view" and index + 1 < len(argv):
            index += 1
            options["web"] = argv[index].startswith("web")
        index += 1
    return options


if __name__ == "__main__":
    import sys

    arguments = _parse_arguments(sys.argv[1:])
    if arguments["web"]:
        ft.run(
            main,
            view=ft.AppView.WEB_BROWSER,
            host=arguments["host"],
            port=arguments["port"],
        )
    else:
        # view=ft.AppView.FLET_APP — настольное окно (по умолчанию)
        ft.run(main)
